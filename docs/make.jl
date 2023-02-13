push!(LOAD_PATH, "../src")
using Documenter, AutoSymPTR

Documenter.HTML(
    mathengine = MathJax3(Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"],
        ),
    )),
)

makedocs(
    sitename="AutoSymPTR.jl",
    modules=[AutoSymPTR],
    pages = [
        "Home" => "index.md",
        "Manual" => "methods.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/AutoSymPTR.jl.git",
)