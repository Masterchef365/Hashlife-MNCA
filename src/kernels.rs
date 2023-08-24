use std::num::NonZeroUsize;

use eframe::epaint::ahash::{AHashSet, AHashMap};
use egui::epaint::ahash::HashMap;
use lru::LruCache;

use crate::{
    array2d::Array2D,
    sim::{calc_block_width, Block, Kernel, KernelResult},
};

pub struct Life;

impl Kernel for Life {
    fn order(&self) -> usize {
        1
    }

    fn exec(&mut self, blocks: [Block; 4]) -> (Block, KernelResult) {
        // Collect everything into a dense buffer
        // TODO: Don't allocate in hot loops lol
        let mut buf: Array2D<u8> = Array2D::new(4, 4);

        for j in 0..2 {
            for i in 0..2 {
                let block = &blocks[i + 2 * j];

                for x in 0..2 {
                    for y in 0..2 {
                        buf[(x + i * 2, y + j * 2)] = u8::from(block[(x, y)]);
                    }
                }
            }
        }

        let mut out_data = vec![false; 4];

        for ((ox, oy), out) in [(0, 0), (1, 0), (0, 1), (1, 1)]
            .into_iter()
            .zip(&mut out_data)
        {
            let mut neighbors = 0;
            let mut center = 0;
            for i in 0..3 {
                for j in 0..3 {
                    let p = (i + ox, j + oy);
                    if (i, j) != (1, 1) {
                        neighbors += buf[p];
                    } else {
                        center = buf[p];
                    }
                }
            }

            *out = if center == 1 {
                matches!(neighbors, 2 | 3)
            } else {
                matches!(neighbors, 3)
            };
        }

        let out_block = Array2D::from_array(2, out_data);

        (out_block, KernelResult::NewBlock)
    }
}

pub struct LayeredKernel {
    /// Given the center cell a number of neighbors overlapping the
    /// "live" cells of each, this function returns next state of the center cell
    decider: fn(bool, &[u16]) -> bool,
    /// Masks from which to interpret layers
    layers: Vec<Array2D<bool>>,
    block_order: usize,
}

impl LayeredKernel {
    pub fn new(decider: fn(bool, &[u16]) -> bool, layers: Vec<Array2D<bool>>) -> Self {
        let dims = (layers[0].width(), layers[0].height());
        assert!(
            layers.iter().all(|l| (l.width(), l.height()) == dims),
            "All kernel layers must be the same size"
        );

        let (width, _height) = dims;
        let block_order = calculate_block_order_from_kernel_width(width);
        dbg!(block_order);

        Self {
            decider,
            layers,
            block_order,
        }
    }
}

impl Kernel for LayeredKernel {
    fn order(&self) -> usize {
        self.block_order
    }

    fn exec(&mut self, blocks: [Block; 4]) -> (Block, KernelResult) {
        let w = calc_block_width(&*self);
        assert_eq!(w, blocks[0].width());

        // Copy everything into a 2D buffer of u8s to make this easier
        let mut buf: Array2D<bool> = Array2D::new(w * 2, w * 2);

        // For each block
        for i in 0..2 {
            for j in 0..2 {
                let block = &blocks[i + 2 * j];

                // For each pixel
                for x in 0..w {
                    for y in 0..w {
                        buf[(x + i * w, y + j * w)] = block[(x, y)];
                    }
                }
            }
        }

        // Now calculate the counts by using a sliding window
        let mut out_data = vec![];
        for j in 0..w {
            for i in 0..w {
                let mut counts = vec![0_u16; self.layers.len()];
                let center = (
                    self.layers[0].width() / 2 + i,
                    self.layers[0].height() / 2 + j,
                );
                for (layer, count) in self.layers.iter().zip(&mut counts) {
                    for y in 0..layer.height() {
                        for x in 0..layer.width() {
                            if layer[(x, y)] && buf[(i + x, j + y)] {
                                *count += 1;
                            }
                        }
                    }
                }

                let result = (self.decider)(buf[center], &counts);
                out_data.push(result);
            }
        }

        let out_block = Array2D::from_array(w, out_data);

        (out_block, KernelResult::NewBlock)
    }
}

/// Given a kernel's width, decide the appropriate block order
/// Returns None if the width is invalid
fn calculate_block_order_from_kernel_width(kernel_width: usize) -> usize {
    for k in 0..=usize::BITS as usize {
        let radius = 1 << k;
        let expected_width = 2 * radius + 1;
        if kernel_width == expected_width {
            let order = k + 1;
            return order;
        }
    }

    panic!("Invalid kernel size, must be >=3x3 and <=2049x2049")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_order() {
        assert_eq!(calculate_block_order_from_kernel_width(3), 1);
        assert_eq!(calculate_block_order_from_kernel_width(5), 2);
        assert_eq!(calculate_block_order_from_kernel_width(9), 3);
        assert_eq!(calculate_block_order_from_kernel_width(17), 4);
        assert_eq!(calculate_block_order_from_kernel_width(33), 5);
    }

    #[test]
    #[should_panic]
    fn test_block_order_panic() {
        calculate_block_order_from_kernel_width(8);
    }

    #[test]
    #[should_panic]
    fn test_block_order_panic2() {
        calculate_block_order_from_kernel_width(usize::MAX);
    }

    #[test]
    #[should_panic]
    fn test_block_order_panic3() {
        calculate_block_order_from_kernel_width(1);
    }
}

pub fn life_layered_kernel() -> LayeredKernel {
    fn decider(center: bool, counts: &[u16]) -> bool {
        let neighbors = counts[0];
        if center {
            matches!(neighbors, 2 | 3)
        } else {
            matches!(neighbors, 3)
        }
    }

    let kernel = [
        [1, 1, 1], // .
        [1, 0, 1], // .
        [1, 1, 1], // .
    ];

    let kernel = kernel
        .into_iter()
        .map(|row| row.into_iter())
        .flatten()
        .map(|i| i == 1)
        .collect();
    let kernel = Array2D::from_array(3, kernel);

    LayeredKernel::new(decider, vec![kernel])
}

pub fn larger_than_life_layered_kernel() -> LayeredKernel {
    fn decider(center: bool, counts: &[u16]) -> bool {
        let neighbors = counts[0];
        let mut output = center;

        if neighbors <= 33 {
            output = false;
        }
        if neighbors >= 34 && neighbors <= 45 {
            output = true;
        }
        if neighbors >= 58 && neighbors <= 121 {
            output = false;
        }

        output
    }

    let kernel = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    let kernel = kernel
        .into_iter()
        .map(|row| row.into_iter())
        .flatten()
        .map(|i| i == 1)
        .collect();

    let kernel = Array2D::from_array(17, kernel);

    print_array(&kernel);

    LayeredKernel::new(decider, vec![kernel])
}

pub fn basic_mnca() -> LayeredKernel {
    let mut layer0 = Array2D::new(17, 17);
    draw_ring(&mut layer0, 5 * 5, 8 * 7);
    print_array(&layer0);
    println!("{}", count_true(&layer0));
    println!();

    let mut layer1 = Array2D::new(17, 17);
    draw_ring(&mut layer1, 1, 3 * 4);
    print_array(&layer1);
    println!("{}", count_true(&layer1));

    fn decider(mut center: bool, counts: &[u16]) -> bool {
        let avg = [counts[0] as f32 / 108.0, counts[1] as f32 / 36.0];

        if avg[0] >= 0.210 && avg[0] <= 0.220 {
            center = true;
        }
        if avg[0] >= 0.350 && avg[0] <= 0.500 {
            center = false;
        }
        if avg[0] >= 0.750 && avg[0] <= 0.850 {
            center = false;
        }
        if avg[1] >= 0.100 && avg[1] <= 0.280 {
            center = false;
        }
        if avg[1] >= 0.430 && avg[1] <= 0.550 {
            center = true;
        }
        if avg[0] >= 0.120 && avg[0] <= 0.150 {
            center = false;
        }

        center
    }

    LayeredKernel::new(decider, vec![layer0, layer1])
}

fn print_array(arr: &Array2D<bool>) {
    for row in arr.data().chunks_exact(arr.width()) {
        for &elem in row {
            if elem {
                print!("# ");
            } else {
                print!("- ");
            }
        }
        println!();
    }
}

fn draw_ring(arr: &mut Array2D<bool>, inner_sq: i32, outer_sq: i32) {
    let w = (arr.width() / 2) as i32;
    for x in -w..=w {
        for y in -w..=w {
            let r2 = x.pow(2) + y.pow(2);
            if r2 >= inner_sq && r2 < outer_sq {
                let i = (x + w) as usize;
                let j = (y + w) as usize;
                arr[(i, j)] = true;
            }
        }
    }
}

fn count_true(arr: &Array2D<bool>) -> usize {
    arr.data().iter().filter(|x| **x).count()
}

pub struct KernelCache {
    cache: AHashMap<Summary, usize>,
    solutions: LruCache<[usize; 4], usize>,
    values: AHashMap<usize, Array2D<bool>>,
    wrap: Box<dyn Kernel>,
    hits: usize,
    next_idx: usize,
}

impl KernelCache {
    pub fn new(wrap: Box<dyn Kernel>) -> Self {
        Self {
            cache: Default::default(),
            solutions: LruCache::new(NonZeroUsize::new(8888).unwrap()),
            values: Default::default(),
            wrap,
            hits: 0,
            next_idx: 0,
        }
    }
}

impl Kernel for KernelCache {
    fn order(&self) -> usize {
        self.wrap.order()
    }

    fn exec(&mut self, blocks: [Block; 4]) -> (Block, KernelResult) {
        const DOWNSAMPLE: usize = 1;
        let hashes = blocks.clone().map(|block| {
            *self
                .cache
                .entry(summarize(&block, DOWNSAMPLE))
                .or_insert_with(|| {
                    let idx = self.next_idx;
                    self.next_idx += 1;
                    self.values.insert(idx, block);
                    idx
                })
        });

        if self.solutions.get(&hashes).is_some() {
            self.hits += 1;
            
            let max_cache = 10_000;
            let max_values = 10_000;
            if self.cache.len() > max_cache || self.values.len() > max_values {
                eprintln!("Garbage collecting");
                let in_cache: AHashSet<usize> = self
                    .solutions
                    .iter()
                    .map(|(&[a, b, c, d], &e)| [a, b, c, d, e])
                    .flatten()
                    .collect();

                self.cache.retain(|_, v| in_cache.contains(v));
                self.values.retain(|k, _| in_cache.contains(k));
                //self.values
                // Garbage collection...
                //self.cache.retain(|_, v|)
            }

            dbg!(
                self.hits,
                self.values.len(),
                self.solutions.len(),
                self.cache.len(),
            );

        }

        let soln_idx = *self.solutions.get_or_insert(hashes, || {
            let (soln, _) = self.wrap.exec(blocks);
            *self
                .cache
                .entry(summarize(&soln, DOWNSAMPLE))
                .or_insert_with(|| {
                    let idx = self.next_idx;
                    self.next_idx += 1;
                    self.values.insert(idx, soln.clone());
                    idx
                })
        });

        let block = self.values.get(&soln_idx).unwrap().clone();

        //dbg!(self.solutions.len());

        (block, KernelResult::NewBlock)
    }
}

type Summary = Array2D<bool>;
//type Summary = usize;

fn summarize(arr: &Array2D<bool>, step: usize) -> Summary {
    arr.clone()
}
