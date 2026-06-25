use std::collections::HashMap;
use std::fmt::Write;

use crate::safetensors::TensorInfo;

/// Format a tensor table with auto-sized columns based on actual content widths.
/// Returns the complete table as a string.
pub fn format_tensor_table(tensors: &HashMap<String, TensorInfo>) -> String {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let header = ["Tensor", "Dtype", "Shape", "Offsets"];

    // Pre-format each row's cells so we can measure widths
    let rows: Vec<[String; 4]> = names
        .iter()
        .map(|name| {
            let info = &tensors[*name];
            let [start, end] = info.data_offsets;
            [
                name.to_string(),
                format!("{:?}", info.dtype),
                format!("{:?}", info.shape),
                format!("{}..{}", start, end),
            ]
        })
        .collect();

    // Compute max width for each column (header vs. data)
    let mut widths = header.map(|h| h.len());
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    let mut output = String::new();

    // Header: first column left-aligned, rest right-aligned
    writeln!(
        output,
        "{:<w0$}  {:>w1$}  {:>w2$}  {:>w3$}",
        header[0],
        header[1],
        header[2],
        header[3],
        w0 = widths[0],
        w1 = widths[1],
        w2 = widths[2],
        w3 = widths[3],
    )
    .unwrap();

    // Separator line
    let total_width = widths.iter().sum::<usize>() + 6;
    writeln!(output, "{}", "-".repeat(total_width)).unwrap();

    // Data rows: first column left-aligned, rest right-aligned
    for row in &rows {
        writeln!(
            output,
            "{:<w0$}  {:>w1$}  {:>w2$}  {:>w3$}",
            row[0],
            row[1],
            row[2],
            row[3],
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
            w3 = widths[3],
        )
        .unwrap();
    }

    // Summary
    writeln!(output, "\nTotal: {} tensors", tensors.len()).unwrap();

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::Dtype;

    fn make_tensor(dtype: Dtype, shape: Vec<usize>, offsets: [usize; 2]) -> TensorInfo {
        TensorInfo {
            dtype,
            shape,
            data_offsets: offsets,
        }
    }

    #[test]
    fn columns_auto_size_to_short_names() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            make_tensor(Dtype::F32, vec![3, 4], [0, 48]),
        );
        tensors.insert("b".to_string(), make_tensor(Dtype::F32, vec![4], [48, 64]));

        let output = format_tensor_table(&tensors);
        let lines: Vec<&str> = output.lines().collect();

        // "Tensor" (6) > "w" (1), so col 0 = 6
        // "Dtype" (5) > "F32" (3), so col 1 = 5
        // "[3, 4]" (6) > "Shape" (5), so col 2 = 6
        // "Offsets" (7) > "48..64" (6), so col 3 = 7
        assert_eq!(lines[0], "Tensor  Dtype   Shape  Offsets");
        assert_eq!(lines[1], "------------------------------");
        assert_eq!(lines[2], "b         F32     [4]   48..64");
        assert_eq!(lines[3], "w         F32  [3, 4]    0..48");
        assert_eq!(lines[5], "Total: 2 tensors");
    }

    #[test]
    fn columns_widen_for_long_tensor_names() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            make_tensor(Dtype::BF16, vec![2048, 2048], [0, 8388608]),
        );
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            make_tensor(Dtype::BF16, vec![151936, 2048], [8388608, 931135488]),
        );

        let output = format_tensor_table(&tensors);
        let lines: Vec<&str> = output.lines().collect();

        // col 0 = 38 (longest name), col 1 = 5 ("Dtype"), col 2 = 14 ("[151936, 2048]"), col 3 = 18 ("8388608..931135488")
        let header_expected = format!(
            "{:<38}  {:>5}  {:>14}  {:>18}",
            "Tensor", "Dtype", "Shape", "Offsets"
        );
        assert_eq!(lines[0], header_expected);

        let separator_len = 38 + 5 + 14 + 18 + 6;
        assert_eq!(lines[1].len(), separator_len);
        assert!(lines[1].chars().all(|c| c == '-'));
    }

    #[test]
    fn empty_tensor_map_shows_only_header() {
        let tensors = HashMap::new();
        let output = format_tensor_table(&tensors);
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines[0], "Tensor  Dtype  Shape  Offsets");
        let separator_len = 6 + 5 + 5 + 7 + 6;
        assert_eq!(lines[1].len(), separator_len);
        assert!(lines[1].chars().all(|c| c == '-'));
        assert_eq!(lines[3], "Total: 0 tensors");
    }

    #[test]
    fn rows_are_sorted_alphabetically() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "z_tensor".to_string(),
            make_tensor(Dtype::F32, vec![1], [0, 4]),
        );
        tensors.insert(
            "a_tensor".to_string(),
            make_tensor(Dtype::F32, vec![2], [4, 12]),
        );
        tensors.insert(
            "m_tensor".to_string(),
            make_tensor(Dtype::F32, vec![3], [12, 24]),
        );

        let output = format_tensor_table(&tensors);
        let lines: Vec<&str> = output.lines().collect();

        assert!(lines[2].starts_with("a_tensor"));
        assert!(lines[3].starts_with("m_tensor"));
        assert!(lines[4].starts_with("z_tensor"));
    }

    #[test]
    fn all_rows_have_same_length() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "short".to_string(),
            make_tensor(Dtype::F32, vec![1], [0, 4]),
        );
        tensors.insert(
            "a_very_long_tensor_name_that_exceeds_everything".to_string(),
            make_tensor(Dtype::BF16, vec![1024, 1024, 3], [4, 12582916]),
        );

        let output = format_tensor_table(&tensors);
        let lines: Vec<&str> = output.lines().collect();

        let expected_len = lines[0].len();
        assert_eq!(lines[1].len(), expected_len);
        assert_eq!(lines[2].len(), expected_len);
        assert_eq!(lines[3].len(), expected_len);
    }
}
