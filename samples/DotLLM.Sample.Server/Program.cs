var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () => "dotLLM Server Sample");

app.Run();
