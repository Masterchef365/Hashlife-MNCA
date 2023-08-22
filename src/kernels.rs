use crate::sim::{Kernel, KernelResult, Block};

pub struct Life;

impl Kernel for Life {
    fn order(&self) -> usize {
        1
    }

    fn approximate(&mut self, blocks: [Block; 4]) -> (Block, KernelResult) {
        // Collect everything into a dense buffer
        let mut buf = [0u8; 16];
        buf
            .iter_mut()
            .zip(blocks.iter().map(|Block(data)| data.iter()).flatten())
            .for_each(|(f, k)| *f = u8::from(*k));

        let mut out_data = vec![false; 4];

        for offset in [0, 1, 4, 5] {
            let mut neighbors = 0;
            let mut center = 0;
            for i in 0..3 {
                for j in 0..3 {
                    let idx = offset + i + j * 4;
                    if (i, j) != (1, 1) {
                        neighbors += buf[idx];
                    } else {
                        center = buf[idx];
                    }
                }
            }

            let result = if center == 1 {
                matches!(neighbors, 2 | 3)
            } else {
                matches!(neighbors, 3)
            };
            out_data.push(result);
        }

        let out_block = Block::new(&*self, out_data);

        (out_block, KernelResult::NewBlock)
    }
}

