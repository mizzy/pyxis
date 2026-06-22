use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: pyxis <model.safetensors>");
    let safetensors = pyxis::safetensors::SafeTensors::load(Path::new(&path))
        .expect("Failed to load safetensors file");

    print!(
        "{}",
        pyxis::display::format_tensor_table(safetensors.tensors())
    );
}
