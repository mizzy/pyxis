pub struct KvCache {
    layers: Vec<LayerCache>,
}

pub struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    kv_dim: usize,
    cached_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, kv_dim: usize) -> Self {
        let layers = (0..num_layers).map(|_| LayerCache::new(kv_dim)).collect();

        Self { layers }
    }

    pub fn layer_mut(&mut self, layer_idx: usize) -> &mut LayerCache {
        &mut self.layers[layer_idx]
    }
}

impl LayerCache {
    pub fn new(kv_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            kv_dim,
            cached_len: 0,
        }
    }

    pub fn append(&mut self, new_keys: &[f32], new_values: &[f32], num_new: usize) {
        assert_eq!(new_keys.len(), num_new * self.kv_dim);
        assert_eq!(new_values.len(), num_new * self.kv_dim);

        self.keys.extend_from_slice(new_keys);
        self.values.extend_from_slice(new_values);
        self.cached_len += num_new;
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.cached_len
    }

    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_cache() {
        let mut cache = KvCache::new(3, 4);

        assert_eq!(cache.layer_mut(0).len(), 0);
        assert_eq!(cache.layer_mut(1).len(), 0);
        assert_eq!(cache.layer_mut(2).len(), 0);
    }

    #[test]
    fn append_grows_cache() {
        let mut cache = LayerCache::new(4);

        cache.append(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[9.0; 8], 2);
        assert_eq!(cache.len(), 2);

        cache.append(&[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0], 1);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn keys_returns_all_cached_data() {
        let mut cache = LayerCache::new(2);

        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2);
        cache.append(&[9.0, 10.0], &[11.0, 12.0], 1);

        assert_eq!(cache.keys(), &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]);
    }
}
