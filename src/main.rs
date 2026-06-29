use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: pyxis <model-dir> [prompt] [--bench] [--quantize] [--quantize-int4] [--metal]"
        );
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let bench_mode = args.iter().any(|arg| arg == "--bench");
    let use_metal = args.iter().any(|arg| arg == "--metal");
    let quantize = if args.iter().any(|arg| arg == "--quantize-int4") {
        Some("int4")
    } else if args.iter().any(|arg| arg == "--quantize") {
        Some("int8")
    } else {
        None
    };
    let prompt = args
        .iter()
        .skip(2)
        .find(|arg| {
            !matches!(
                arg.as_str(),
                "--bench" | "--quantize" | "--quantize-int4" | "--metal"
            )
        })
        .map_or("Hello", |s| s.as_str());

    let load_start = std::time::Instant::now();
    let model = pyxis::model::Model::load(Path::new(model_dir), quantize, use_metal)
        .expect("Failed to load model");
    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if bench_mode {
        let mut result = model.benchmark(prompt, 20);
        result.load_time_ms = load_time_ms;
        result.display();
    } else {
        model.generate(prompt, 50);
        println!();
    }
}
