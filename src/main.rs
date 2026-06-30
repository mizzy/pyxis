use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: pyxis <model-dir> [prompt] [--bench] [--quantize] [--quantize-int4] [--metal] [--show-tensors]"
        );
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let bench_mode = args.iter().any(|arg| arg == "--bench");
    let use_metal = args.iter().any(|arg| arg == "--metal");
    let show_tensors = args.iter().any(|arg| arg == "--show-tensors");
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
                "--bench" | "--quantize" | "--quantize-int4" | "--metal" | "--show-tensors"
            )
        })
        .map_or("Hello", |s| s.as_str());

    let model_path = Path::new(model_dir);
    if show_tensors {
        if pyxis::model::is_gguf_path(model_path) {
            let gguf = pyxis::gguf::GgufFile::parse(model_path).expect("Failed to parse GGUF file");
            eprint!(
                "{}",
                pyxis::display::format_gguf_tensor_table(&gguf.tensors)
            );
        } else {
            let store = pyxis::safetensors::TensorStore::load(model_path)
                .expect("Failed to load safetensors");
            eprint!("{}", pyxis::display::format_tensor_table(&store.tensors()));
        }
        return;
    }

    let load_start = std::time::Instant::now();
    let model =
        pyxis::model::Model::load(model_path, quantize, use_metal).expect("Failed to load model");
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
