use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: pyxis <model-dir> [prompt]");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let prompt = args.get(2).map_or("Hello", |s| s.as_str());

    let model = pyxis::model::Model::load(Path::new(model_dir)).expect("Failed to load model");
    model.generate(prompt, 50);
    println!();
}
