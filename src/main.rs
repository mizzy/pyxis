use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: pyxis <model-dir> [prompt] [--bench]");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let bench_mode = args.iter().any(|arg| arg == "--bench");
    let prompt = args
        .iter()
        .skip(2)
        .find(|arg| *arg != "--bench")
        .map_or("Hello", |s| s.as_str());

    let load_start = std::time::Instant::now();
    let model = pyxis::model::Model::load(Path::new(model_dir)).expect("Failed to load model");
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
