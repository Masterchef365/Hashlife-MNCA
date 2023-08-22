use crate::{sim::{Kernel, KernelResult, Block}, array2d::Array2D};

pub struct Life;

impl Kernel for Life {
    fn order(&self) -> usize {
        1
    }

    fn approximate(&mut self, blocks: [Block; 4]) -> (Block, KernelResult) {
        // Collect everything into a dense buffer
        // TODO: Don't allocate in hot loops lol
        let mut buf: Array2D<u8> = Array2D::new(4, 4);

        for i in 0..2 {
            for j in 0..2 {
                let block = &blocks[i + j*2];
                for x in 0..2 {
                    for y in 0..2 {
                        buf[(x + i * 2, y + j * 2)] = u8::from(block[(x, y)]);
                    }
                }
            }
        }

        let mut out_data = vec![false; 4];

        for ((ox, oy), out) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().zip(&mut out_data) {
            let mut neighbors = 0;
            let center = buf[(ox + 1, oy + 1)];
            for i in 0..3 {
                for j in 0..3 {
                    if (i, j) != (1, 1) {
                        neighbors += buf[(i + ox, j + oy)];
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

