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

    LayeredKernel::new(decider, vec![kernel])
}
