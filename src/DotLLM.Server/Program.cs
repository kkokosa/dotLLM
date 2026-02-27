var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () => "dotLLM Server");

app.Run();
