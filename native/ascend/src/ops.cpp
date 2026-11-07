#include "dotllm_ascend.h"

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_rotary_position_embedding.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_softmax.h>
#include <cmath>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
thread_local char last_error[2048] = {};

int32_t fail(int32_t status, const char* operation) noexcept {
    const char* detail = aclGetRecentErrMsg();
    std::snprintf(last_error, sizeof(last_error), "%s: status=%d; %s",
                  operation, status, detail ? detail : "no CANN diagnostic");
    return status;
}

struct TensorDeleter {
    void operator()(aclTensor* tensor) const noexcept {
        if (tensor) aclDestroyTensor(tensor);
    }
};
using Tensor = std::unique_ptr<aclTensor, TensorDeleter>;

struct Workspace {
    void* data = nullptr;
    ~Workspace() { if (data) aclrtFree(data); }
};

struct Executor {
    aclOpExecutor* data = nullptr;
    // Immediately mark the executor repeatable to give this scope explicit ownership,
    // including workspace-allocation failure. Execute once; never cache by shape.
    ~Executor() { if (data) aclDestroyAclOpExecutor(data); }
};

bool valid_matrix(int64_t rows, int64_t columns) noexcept {
    return rows > 0 && columns > 0 &&
           rows <= std::numeric_limits<int64_t>::max() / 4 / columns;
}

struct NativeFailure { int32_t status; };
void check(int32_t status, const char* operation) {
    if (status != 0) throw NativeFailure{fail(status, operation)};
}

void require(bool condition) {
    if (!condition) throw std::invalid_argument("Invalid pointers, shape, or operation parameters.");
}

struct Buffer {
    void* data = nullptr;
    explicit Buffer(int64_t bytes) {
        require(bytes > 0);
        check(aclrtMalloc(&data, static_cast<size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~Buffer() { if (data) aclrtFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Scalar {
    aclScalar* data;
    explicit Scalar(float value) : data(aclCreateScalar(&value, ACL_FLOAT)) {
        require(data != nullptr);
    }
    ~Scalar() { aclDestroyScalar(data); }
};

Tensor tensor(const void* data, aclDataType dtype, std::initializer_list<int64_t> dimensions,
              std::initializer_list<int64_t> strides = {},
              std::initializer_list<int64_t> storage = {}, int64_t offset = 0) {
    std::vector<int64_t> shape(dimensions), stride(strides), stored(storage);
    if (stride.empty()) {
        stride.resize(shape.size());
        int64_t step = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            require(valid_matrix(step, shape[i]));
            stride[i] = step;
            step *= shape[i];
        }
    }
    if (stored.empty()) stored = shape;
    Tensor result(aclCreateTensor(shape.data(), shape.size(), dtype, stride.data(), offset,
        ACL_FORMAT_ND, stored.data(), stored.size(), const_cast<void*>(data)));
    if (!result) check(-2, "aclCreateTensor");
    return result;
}

template <typename Operation>
int32_t protect(Operation operation) noexcept {
    last_error[0] = '\0';
    try {
        operation();
        return 0;
    } catch (const NativeFailure& error) {
        return error.status;
    } catch (const std::exception& error) {
        std::snprintf(last_error, sizeof(last_error), "%s", error.what());
        return -3;
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "Unknown C++ exception.");
        return -3;
    }
}

// Repeatable executors remain caller-owned, even after one execution.
// Synchronize before releasing their workspace and all input/output descriptors.
template <typename Run>
void execute(uint64_t bytes, aclOpExecutor* planned, aclrtStream stream, Run run, const char* name) {
    require(planned != nullptr);
    Executor executor{planned};
    check(aclSetAclOpExecutorRepeatable(executor.data), "aclSetAclOpExecutorRepeatable");
    Workspace workspace;
    if (bytes) check(aclrtMalloc(&workspace.data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
    int32_t status = run(workspace.data, bytes, executor.data, stream);
    if (status != 0) {
        fail(status, name);
        aclrtSynchronizeStream(stream);
        throw NativeFailure{status};
    }
    check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
}

#define OP(name, ...) do { \
    uint64_t workspace_bytes = 0; \
    aclOpExecutor* planned_op = nullptr; \
    check(aclnn##name##GetWorkspaceSize(__VA_ARGS__, &workspace_bytes, &planned_op), \
          "aclnn" #name "GetWorkspaceSize"); \
    execute(workspace_bytes, planned_op, stream, aclnn##name, "aclnn" #name); \
} while (false)

void cast(const aclTensor* source, aclDataType dtype, aclTensor* destination, aclrtStream stream) {
    OP(Cast, source, dtype, destination);
}
} // namespace

extern "C" uint32_t dotllm_ascend_abi_version(void) { return 3; }
extern "C" const char* dotllm_ascend_last_error(void) { return last_error; }

extern "C" int32_t dotllm_ascend_linear_f16(
    const void* x, const void* weight, void* output,
    int64_t m, int64_t n, int64_t k, void* stream) {
    last_error[0] = '\0';
    if (!x || !weight || !output || !stream || output == x || output == weight ||
        !valid_matrix(m, k) || !valid_matrix(n, k) || !valid_matrix(m, n)) {
        std::snprintf(last_error, sizeof(last_error), "Invalid Linear pointers or M/N/K.");
        return -1;
    }
    try {
        const int64_t x_shape[] = {m, k}, x_stride[] = {k, 1};
        const int64_t w_storage[] = {n, k};
        // Transposed view of row-major W[N,K]. Strides are in elements, not bytes.
        const int64_t w_shape[] = {k, n}, w_stride[] = {1, k};
        const int64_t y_shape[] = {m, n}, y_stride[] = {n, 1};
        Tensor xt(aclCreateTensor(x_shape, 2, ACL_FLOAT16, x_stride, 0,
                                 ACL_FORMAT_ND, x_shape, 2, const_cast<void*>(x)));
        Tensor wt(aclCreateTensor(w_shape, 2, ACL_FLOAT16, w_stride, 0,
                                 ACL_FORMAT_ND, w_storage, 2, const_cast<void*>(weight)));
        Tensor yt(aclCreateTensor(y_shape, 2, ACL_FLOAT16, y_stride, 0,
                                 ACL_FORMAT_ND, y_shape, 2, output));
        if (!xt || !wt || !yt) return fail(-2, "aclCreateTensor");

        uint64_t workspace_size = 0;
        aclOpExecutor* planned = nullptr;
        // KEEP_DTYPE does not request reduced-precision FP32 input conversion.
        int32_t status = aclnnMatmulGetWorkspaceSize(
            xt.get(), wt.get(), yt.get(), 0, &workspace_size, &planned);
        if (status != 0) return fail(status, "aclnnMatmulGetWorkspaceSize");
        Executor executor{planned};
        status = aclSetAclOpExecutorRepeatable(executor.data);
        if (status != 0) return fail(status, "aclSetAclOpExecutorRepeatable");

        Workspace workspace;
        if (workspace_size != 0) {
            status = aclrtMalloc(&workspace.data, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
            if (status != 0) return fail(status, "aclrtMalloc(workspace)");
        }
        status = aclnnMatmul(workspace.data, workspace_size, executor.data, stream);
        if (status != 0) {
            fail(status, "aclnnMatmul");
            // A failed enqueue can still have submitted work. Retain all storage until drained.
            aclrtSynchronizeStream(stream);
            return status;
        }
        status = aclrtSynchronizeStream(stream);
        if (status != 0) return fail(status, "aclrtSynchronizeStream");
        return 0;
    } catch (const std::exception& error) {
        std::snprintf(last_error, sizeof(last_error), "C++ exception: %s", error.what());
        return -3;
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "Unknown C++ exception.");
        return -3;
    }
}

extern "C" int32_t dotllm_ascend_rms_norm_f16(
    const void* x, const void* weight, void* output,
    int64_t rows, int64_t width, double epsilon, void* stream) {
    return protect([&] {
        require(x && weight && output && stream && output != x && output != weight &&
                valid_matrix(rows, width) && std::isfinite(epsilon) && epsilon > 0);
        auto xt = tensor(x, ACL_FLOAT16, {rows, width});
        auto wt = tensor(weight, ACL_FLOAT16, {width});
        auto yt = tensor(output, ACL_FLOAT16, {rows, width});
        Buffer rstd(rows * 4);
        auto rt = tensor(rstd.data, ACL_FLOAT, {rows, 1});
        OP(RmsNorm, xt.get(), wt.get(), epsilon, yt.get(), rt.get());
    });
}

extern "C" int32_t dotllm_ascend_add_f16(
    const void* left, const void* right, void* output,
    int64_t rows, int64_t width, int64_t right_rows, void* stream) {
    return protect([&] {
        require(left && right && output && stream && output != left && output != right &&
                valid_matrix(rows, width) && (right_rows == 1 || right_rows == rows));
        auto lt = tensor(left, ACL_FLOAT16, {rows, width});
        auto rt = tensor(right, ACL_FLOAT16, {right_rows, width});
        auto yt = tensor(output, ACL_FLOAT16, {rows, width});
        Scalar one(1);
        OP(Add, lt.get(), rt.get(), one.data, yt.get());
    });
}

extern "C" int32_t dotllm_ascend_swiglu_f16(
    const void* gate, const void* up, void* output, int64_t count, void* stream) {
    return protect([&] {
        require(gate && up && output && stream && output != gate && output != up && valid_matrix(count, 1));
        auto gt = tensor(gate, ACL_FLOAT16, {count});
        auto ut = tensor(up, ACL_FLOAT16, {count});
        auto yt = tensor(output, ACL_FLOAT16, {count});
        Buffer gf(count * 4), uf(count * 4), silu(count * 4), product(count * 4);
        auto gft = tensor(gf.data, ACL_FLOAT, {count});
        auto uft = tensor(uf.data, ACL_FLOAT, {count});
        auto st = tensor(silu.data, ACL_FLOAT, {count});
        auto pt = tensor(product.data, ACL_FLOAT, {count});
        cast(gt.get(), ACL_FLOAT, gft.get(), stream);
        cast(ut.get(), ACL_FLOAT, uft.get(), stream);
        OP(Silu, gft.get(), st.get());
        OP(Mul, st.get(), uft.get(), pt.get());
        cast(pt.get(), ACL_FLOAT16, yt.get(), stream);
    });
}

extern "C" int32_t dotllm_ascend_rope_f16(
    const void* x, const void* cos, const void* sin, void* output,
    int64_t rows, int64_t heads, int64_t head_dim, int64_t rotary_dim,
    int32_t neox, void* stream) {
    return protect([&] {
        require(x && cos && sin && output && stream && output != x && output != cos && output != sin &&
                valid_matrix(heads, head_dim) && valid_matrix(rows, heads * head_dim) &&
                rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0 && (neox == 0 || neox == 1));
        const int64_t row_width = heads * head_dim;
        auto xt = tensor(x, ACL_FLOAT16, {1, rows, heads, rotary_dim},
            {rows * row_width, row_width, head_dim, 1}, {1, rows, heads, head_dim});
        auto ct = tensor(cos, ACL_FLOAT16, {1, rows, 1, rotary_dim});
        auto st = tensor(sin, ACL_FLOAT16, {1, rows, 1, rotary_dim});
        // Use contiguous FP32 inputs, as the CANN reference backend does, to avoid
        // reduced-precision/strided-input behavior in the fused rotary kernel.
        const int64_t count = rows * heads * rotary_dim;
        Buffer xf(count * 4), cf(rows * rotary_dim * 4), sf(rows * rotary_dim * 4),
            yf(count * 4), rotated(count * 2);
        auto xft = tensor(xf.data, ACL_FLOAT, {1, rows, heads, rotary_dim});
        auto cft = tensor(cf.data, ACL_FLOAT, {1, rows, 1, rotary_dim});
        auto sft = tensor(sf.data, ACL_FLOAT, {1, rows, 1, rotary_dim});
        auto yft = tensor(yf.data, ACL_FLOAT, {1, rows, heads, rotary_dim});
        auto yt = tensor(rotated.data, ACL_FLOAT16, {1, rows, heads, rotary_dim});
        cast(xt.get(), ACL_FLOAT, xft.get(), stream);
        cast(ct.get(), ACL_FLOAT, cft.get(), stream);
        cast(st.get(), ACL_FLOAT, sft.get(), stream);
        // CANN mode 1 matches GGUF interleaved pairs; mode 0 matches NeoX halves.
        OP(RotaryPositionEmbedding, xft.get(), cft.get(), sft.get(), neox ? 0 : 1, yft.get());
        cast(yft.get(), ACL_FLOAT16, yt.get(), stream);
        const int64_t total_bytes = rows * row_width * 2;
        if (rotary_dim == head_dim) {
            check(aclrtMemcpy(output, total_bytes, rotated.data, total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE copy");
        } else {
            check(aclrtMemcpy(output, total_bytes, x, total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE tail copy");
            for (int64_t i = 0; i < rows * heads; ++i)
                check(aclrtMemcpy(static_cast<char*>(output) + i * head_dim * 2, rotary_dim * 2,
                    static_cast<char*>(rotated.data) + i * rotary_dim * 2, rotary_dim * 2,
                    ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE head copy");
        }
    });
}

extern "C" int32_t dotllm_ascend_attention_f16(
    const void* query, const void* keys, const void* values, void* output,
    int64_t rows, int64_t kv_rows, int64_t heads, int64_t kv_heads,
    int64_t head_dim, int64_t query_offset, void* stream) {
    return protect([&] {
        require(query && keys && values && output && stream &&
                output != query && output != keys && output != values &&
                valid_matrix(heads, head_dim) && kv_heads > 0 && heads % kv_heads == 0 &&
                valid_matrix(rows, heads * head_dim) && valid_matrix(kv_rows, kv_heads * head_dim) &&
                valid_matrix(rows, kv_rows) && query_offset >= 0 && rows <= kv_rows &&
                query_offset <= kv_rows - rows);
        const int64_t q_width = heads * head_dim, kv_width = kv_heads * head_dim;
        Buffer qf(rows * q_width * 4), kf(kv_rows * kv_width * 4), vf(kv_rows * kv_width * 4);
        auto qh = tensor(query, ACL_FLOAT16, {rows, q_width});
        auto kh = tensor(keys, ACL_FLOAT16, {kv_rows, kv_width});
        auto vh = tensor(values, ACL_FLOAT16, {kv_rows, kv_width});
        auto qt = tensor(qf.data, ACL_FLOAT, {rows, q_width});
        auto kt = tensor(kf.data, ACL_FLOAT, {kv_rows, kv_width});
        auto vt = tensor(vf.data, ACL_FLOAT, {kv_rows, kv_width});
        cast(qh.get(), ACL_FLOAT, qt.get(), stream);
        cast(kh.get(), ACL_FLOAT, kt.get(), stream);
        cast(vh.get(), ACL_FLOAT, vt.get(), stream);

        // Only control metadata is constructed on the host. Q/K/V, scores, softmax,
        // and value accumulation stay on the NPU throughout the operation.
        std::vector<float> mask(static_cast<size_t>(rows * kv_rows), 0);
        for (int64_t row = 0; row < rows; ++row)
            for (int64_t column = query_offset + row + 1; column < kv_rows; ++column)
                mask[row * kv_rows + column] = -std::numeric_limits<float>::infinity();
        Buffer mask_device(rows * kv_rows * 4), score(rows * kv_rows * 4),
            scaled(rows * kv_rows * 4), masked(rows * kv_rows * 4), probability(rows * kv_rows * 4),
            result(rows * head_dim * 4), result_half(rows * head_dim * 2);
        check(aclrtMemcpy(mask_device.data, rows * kv_rows * 4, mask.data(), rows * kv_rows * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE), "Attention mask upload");
        auto mt = tensor(mask_device.data, ACL_FLOAT, {rows, kv_rows});
        auto score_t = tensor(score.data, ACL_FLOAT, {rows, kv_rows});
        auto scaled_t = tensor(scaled.data, ACL_FLOAT, {rows, kv_rows});
        auto masked_t = tensor(masked.data, ACL_FLOAT, {rows, kv_rows});
        auto prob_t = tensor(probability.data, ACL_FLOAT, {rows, kv_rows});
        auto result_t = tensor(result.data, ACL_FLOAT, {rows, head_dim});
        auto result_h = tensor(result_half.data, ACL_FLOAT16, {rows, head_dim});
        Scalar scale(1.0f / std::sqrt(static_cast<float>(head_dim))), one(1);
        for (int64_t head = 0; head < heads; ++head) {
            int64_t kv_head = head / (heads / kv_heads);
            auto q = tensor(qf.data, ACL_FLOAT,
                {rows, head_dim}, {q_width, 1}, {rows, q_width}, head * head_dim);
            auto k = tensor(kf.data, ACL_FLOAT,
                {head_dim, kv_rows}, {1, kv_width}, {kv_rows, kv_width}, kv_head * head_dim);
            auto v = tensor(vf.data, ACL_FLOAT,
                {kv_rows, head_dim}, {kv_width, 1}, {kv_rows, kv_width}, kv_head * head_dim);
            OP(Matmul, q.get(), k.get(), score_t.get(), 0);
            OP(Muls, score_t.get(), scale.data, scaled_t.get());
            OP(Add, scaled_t.get(), mt.get(), one.data, masked_t.get());
            OP(Softmax, masked_t.get(), -1, prob_t.get());
            OP(Matmul, prob_t.get(), v.get(), result_t.get(), 0);
            cast(result_t.get(), ACL_FLOAT16, result_h.get(), stream);
            for (int64_t row = 0; row < rows; ++row)
                check(aclrtMemcpy(static_cast<char*>(output) + (row * q_width + head * head_dim) * 2,
                    head_dim * 2, static_cast<char*>(result_half.data) + row * head_dim * 2,
                    head_dim * 2, ACL_MEMCPY_DEVICE_TO_DEVICE), "Attention head copy");
        }
    });
}

extern "C" int32_t dotllm_ascend_cast(
    const void* x, void* output, int64_t count, int32_t to_float32, void* stream) {
    return protect([&] {
        require(x && output && stream && x != output && valid_matrix(count, 1) &&
                (to_float32 == 0 || to_float32 == 1));
        auto xt = tensor(x, to_float32 ? ACL_FLOAT16 : ACL_FLOAT, {count});
        auto yt = tensor(output, to_float32 ? ACL_FLOAT : ACL_FLOAT16, {count});
        cast(xt.get(), to_float32 ? ACL_FLOAT : ACL_FLOAT16, yt.get(), stream);
    });
}

extern "C" int32_t dotllm_ascend_linear_f32(
    const void* x, const void* weight, void* output,
    int64_t rows, int64_t n, int64_t k, void* stream) {
    return protect([&] {
        require(x && weight && output && stream && output != x && output != weight &&
                valid_matrix(rows, k) && valid_matrix(n, k) && valid_matrix(rows, n));
        auto xt = tensor(x, ACL_FLOAT, {rows, k});
        auto wt = tensor(weight, ACL_FLOAT, {k, n}, {1, k}, {n, k});
        auto yt = tensor(output, ACL_FLOAT, {rows, n});
        OP(Matmul, xt.get(), wt.get(), yt.get(), 0);
    });
}

extern "C" int32_t dotllm_ascend_rms_norm_f32(
    const void* x, const void* weight, void* output,
    int64_t rows, int64_t width, double epsilon, void* stream) {
    return protect([&] {
        require(x && weight && output && stream && output != x && output != weight &&
                valid_matrix(rows, width) && std::isfinite(epsilon) && epsilon > 0);
        auto xt = tensor(x, ACL_FLOAT, {rows, width});
        auto wt = tensor(weight, ACL_FLOAT, {width});
        Buffer rstd(rows * 4);
        auto yt = tensor(output, ACL_FLOAT, {rows, width});
        auto rt = tensor(rstd.data, ACL_FLOAT, {rows, 1});
        OP(RmsNorm, xt.get(), wt.get(), epsilon, yt.get(), rt.get());
    });
}

extern "C" int32_t dotllm_ascend_add_f32(
    const void* left, const void* right, void* output,
    int64_t rows, int64_t width, int64_t right_rows, int32_t right_f16, void* stream) {
    return protect([&] {
        require(left && right && output && stream && output != left && output != right &&
                valid_matrix(rows, width) && (right_rows == 1 || right_rows == rows) &&
                (right_f16 == 0 || right_f16 == 1));
        auto lt = tensor(left, ACL_FLOAT, {rows, width});
        auto rt = tensor(right, right_f16 ? ACL_FLOAT16 : ACL_FLOAT, {right_rows, width});
        std::unique_ptr<Buffer> converted;
        if (right_f16) {
            converted = std::make_unique<Buffer>(right_rows * width * 4);
            auto rf = tensor(converted->data, ACL_FLOAT, {right_rows, width});
            cast(rt.get(), ACL_FLOAT, rf.get(), stream);
            rt = std::move(rf);
        }
        auto yt = tensor(output, ACL_FLOAT, {rows, width});
        Scalar one(1);
        OP(Add, lt.get(), rt.get(), one.data, yt.get());
    });
}

extern "C" int32_t dotllm_ascend_swiglu_f32(
    const void* gate, const void* up, void* output, int64_t count, void* stream) {
    return protect([&] {
        require(gate && up && output && stream && output != gate && output != up && valid_matrix(count, 1));
        auto gt = tensor(gate, ACL_FLOAT, {count});
        auto ut = tensor(up, ACL_FLOAT, {count});
        auto yt = tensor(output, ACL_FLOAT, {count});
        Buffer silu(count * 4);
        auto st = tensor(silu.data, ACL_FLOAT, {count});
        OP(Silu, gt.get(), st.get());
        OP(Mul, st.get(), ut.get(), yt.get());
    });
}
extern "C" int32_t dotllm_ascend_rope_f32(
    const void* x, const void* cos, const void* sin, void* output,
    int64_t rows, int64_t heads, int64_t head_dim, int64_t rotary_dim,
    int32_t neox, void* stream) {
    return protect([&] {
        require(x && cos && sin && output && stream && output != x && output != cos && output != sin &&
                valid_matrix(heads, head_dim) && valid_matrix(rows, heads * head_dim) &&
                rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0 && (neox == 0 || neox == 1));
        const int64_t row_width = heads * head_dim;
        auto xt = tensor(x, ACL_FLOAT, {1, rows, heads, rotary_dim},
            {rows * row_width, row_width, head_dim, 1}, {1, rows, heads, head_dim});
        auto ct = tensor(cos, ACL_FLOAT16, {1, rows, 1, rotary_dim});
        auto st = tensor(sin, ACL_FLOAT16, {1, rows, 1, rotary_dim});
        // Use contiguous FP32 inputs, as the CANN reference backend does, to avoid
        // reduced-precision/strided-input behavior in the fused rotary kernel.
        const int64_t count = rows * heads * rotary_dim;
        Buffer xf(count * 4), cf(rows * rotary_dim * 4), sf(rows * rotary_dim * 4),
            yf(count * 4), rotated(count * 4);
        auto xft = tensor(xf.data, ACL_FLOAT, {1, rows, heads, rotary_dim});
        auto cft = tensor(cf.data, ACL_FLOAT, {1, rows, 1, rotary_dim});
        auto sft = tensor(sf.data, ACL_FLOAT, {1, rows, 1, rotary_dim});
        auto yft = tensor(yf.data, ACL_FLOAT, {1, rows, heads, rotary_dim});
        auto yt = tensor(rotated.data, ACL_FLOAT, {1, rows, heads, rotary_dim});
        for (int64_t i = 0; i < rows * heads; ++i)
            check(aclrtMemcpy(static_cast<char*>(xf.data) + i * rotary_dim * 4, rotary_dim * 4,
                static_cast<const char*>(x) + i * head_dim * 4, rotary_dim * 4, ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE FP32 gather");
        cast(ct.get(), ACL_FLOAT, cft.get(), stream);
        cast(st.get(), ACL_FLOAT, sft.get(), stream);
        // CANN mode 1 matches GGUF interleaved pairs; mode 0 matches NeoX halves.
        OP(RotaryPositionEmbedding, xft.get(), cft.get(), sft.get(), neox ? 0 : 1, yft.get());
        check(aclrtMemcpy(rotated.data, count * 4, yf.data, count * 4, ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE FP32 result");
        const int64_t total_bytes = rows * row_width * 4;
        if (rotary_dim == head_dim) {
            check(aclrtMemcpy(output, total_bytes, rotated.data, total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE copy");
        } else {
            check(aclrtMemcpy(output, total_bytes, x, total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE tail copy");
            for (int64_t i = 0; i < rows * heads; ++i)
                check(aclrtMemcpy(static_cast<char*>(output) + i * head_dim * 4, rotary_dim * 4,
                    static_cast<char*>(rotated.data) + i * rotary_dim * 4, rotary_dim * 4,
                    ACL_MEMCPY_DEVICE_TO_DEVICE), "RoPE head copy");
        }
    });
}

extern "C" int32_t dotllm_ascend_attention_f32(
    const void* query, const void* keys, const void* values, void* output,
    int64_t rows, int64_t kv_rows, int64_t heads, int64_t kv_heads,
    int64_t head_dim, int64_t query_offset, int32_t kv_f16, void* stream) {
    return protect([&] {
        require(query && keys && values && output && stream &&
                output != query && output != keys && output != values &&
                valid_matrix(heads, head_dim) && kv_heads > 0 && heads % kv_heads == 0 &&
                valid_matrix(rows, heads * head_dim) && valid_matrix(kv_rows, kv_heads * head_dim) &&
                valid_matrix(rows, kv_rows) && query_offset >= 0 && rows <= kv_rows &&
                query_offset <= kv_rows - rows && (kv_f16 == 0 || kv_f16 == 1));
        const int64_t q_width = heads * head_dim, kv_width = kv_heads * head_dim;
        Buffer qf(rows * q_width * 4), kf(kv_rows * kv_width * 4), vf(kv_rows * kv_width * 4);
        auto qh = tensor(query, ACL_FLOAT, {rows, q_width});
        auto kh = tensor(keys, kv_f16 ? ACL_FLOAT16 : ACL_FLOAT, {kv_rows, kv_width});
        auto vh = tensor(values, kv_f16 ? ACL_FLOAT16 : ACL_FLOAT, {kv_rows, kv_width});
        auto qt = tensor(qf.data, ACL_FLOAT, {rows, q_width});
        auto kt = tensor(kf.data, ACL_FLOAT, {kv_rows, kv_width});
        auto vt = tensor(vf.data, ACL_FLOAT, {kv_rows, kv_width});
        check(aclrtMemcpy(qf.data, rows * q_width * 4, query, rows * q_width * 4, ACL_MEMCPY_DEVICE_TO_DEVICE), "Attention FP32 query");
        if (kv_f16) {
            cast(kh.get(), ACL_FLOAT, kt.get(), stream);
            cast(vh.get(), ACL_FLOAT, vt.get(), stream);
        } else {
            check(aclrtMemcpy(kf.data, kv_rows * kv_width * 4, keys, kv_rows * kv_width * 4, ACL_MEMCPY_DEVICE_TO_DEVICE), "Attention FP32 keys");
            check(aclrtMemcpy(vf.data, kv_rows * kv_width * 4, values, kv_rows * kv_width * 4, ACL_MEMCPY_DEVICE_TO_DEVICE), "Attention FP32 values");
        }

        // Only control metadata is constructed on the host. Q/K/V, scores, softmax,
        // and value accumulation stay on the NPU throughout the operation.
        std::vector<float> mask(static_cast<size_t>(rows * kv_rows), 0);
        for (int64_t row = 0; row < rows; ++row)
            for (int64_t column = query_offset + row + 1; column < kv_rows; ++column)
                mask[row * kv_rows + column] = -std::numeric_limits<float>::infinity();
        Buffer mask_device(rows * kv_rows * 4), score(rows * kv_rows * 4),
            scaled(rows * kv_rows * 4), masked(rows * kv_rows * 4), probability(rows * kv_rows * 4),
            result(rows * head_dim * 4), result_half(rows * head_dim * 4);
        check(aclrtMemcpy(mask_device.data, rows * kv_rows * 4, mask.data(), rows * kv_rows * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE), "Attention mask upload");
        auto mt = tensor(mask_device.data, ACL_FLOAT, {rows, kv_rows});
        auto score_t = tensor(score.data, ACL_FLOAT, {rows, kv_rows});
        auto scaled_t = tensor(scaled.data, ACL_FLOAT, {rows, kv_rows});
        auto masked_t = tensor(masked.data, ACL_FLOAT, {rows, kv_rows});
        auto prob_t = tensor(probability.data, ACL_FLOAT, {rows, kv_rows});
        auto result_t = tensor(result.data, ACL_FLOAT, {rows, head_dim});
        auto result_h = tensor(result_half.data, ACL_FLOAT, {rows, head_dim});
        Scalar scale(1.0f / std::sqrt(static_cast<float>(head_dim))), one(1);
        for (int64_t head = 0; head < heads; ++head) {
            int64_t kv_head = head / (heads / kv_heads);
            auto q = tensor(qf.data, ACL_FLOAT,
                {rows, head_dim}, {q_width, 1}, {rows, q_width}, head * head_dim);
            auto k = tensor(kf.data, ACL_FLOAT,
                {head_dim, kv_rows}, {1, kv_width}, {kv_rows, kv_width}, kv_head * head_dim);
            auto v = tensor(vf.data, ACL_FLOAT,
                {kv_rows, head_dim}, {kv_width, 1}, {kv_rows, kv_width}, kv_head * head_dim);
            OP(Matmul, q.get(), k.get(), score_t.get(), 0);
            OP(Muls, score_t.get(), scale.data, scaled_t.get());
            OP(Add, scaled_t.get(), mt.get(), one.data, masked_t.get());
            OP(Softmax, masked_t.get(), -1, prob_t.get());
            OP(Matmul, prob_t.get(), v.get(), result_t.get(), 0);
            check(aclrtMemcpy(result_half.data, rows * head_dim * 4, result.data, rows * head_dim * 4, ACL_MEMCPY_DEVICE_TO_DEVICE), "Attention FP32 result");
            for (int64_t row = 0; row < rows; ++row)
                check(aclrtMemcpy(static_cast<char*>(output) + (row * q_width + head * head_dim) * 4,
                    head_dim * 4, static_cast<char*>(result_half.data) + row * head_dim * 4,
                    head_dim * 4, ACL_MEMCPY_DEVICE_TO_DEVICE), "Attention head copy");
        }
    });
}

