use crate::array2d::Array2D;

/// Block data, whose size is known by the Kernel
pub type Block = Array2D<bool>;

pub trait Kernel {
    /// Power law size of the basic block. E.g. each block has a width of 2^n, where n = self.order()
    /// For HashLife/Conway life (radius 1), this would be 1, since input would be 4 2x2 blocks and output would be 1 2x2 block
    /// For an MNCA with radius 4, this would be 3, since the input would be 4 8x8 blocks and output would be 1 8x8 block
    fn order(&self) -> usize;

    /// Given a novel combination of 4 blocks, produce an output block advanced by one time step
    /// (each entry in the input and output blocks are either 0 or 1 indicating dead or live states respectively)
    fn exec(&mut self, blocks: [Block; 4]) -> (Block, KernelResult);
}

#[derive(Clone, Debug, Copy)]
pub enum KernelResult {
    /// An exact, new block has been returned. This new block should be hashed and inserted into the macrocell tree.
    NewBlock,
    /// An approximate block has been returned. This new block should be hashed and should be found to already be in the macrocell tree.
    Approximate,
}

pub struct Dense {
    back: Array2D<Block>,
    front: Array2D<Block>,
    kernel: Box<dyn Kernel>,
    zero_borders: bool,
}

impl Dense {
    pub fn new(kernel: Box<dyn Kernel>, width: usize, height: usize) -> Self {
        // To account for difference in size between frames, we add 1 to width and height
        let zero_block = Array2D::new(1 << kernel.order(), 1 << kernel.order());
        let zeros = vec![zero_block; (width + 1) * (height + 1)];

        Self {
            front: Array2D::from_array(width + 1, zeros.clone()),
            back: Array2D::from_array(width + 1, zeros),
            kernel,
            zero_borders: true,
        }
    }

    pub fn step(&mut self) {
        for i in 0..self.front.width() as i32 {
            for j in 0..self.front.height() as i32 {
                let (x, y) = if self.zero_borders {
                    (i - 1, j - 1)
                } else {
                    (i, j)
                };

                let in_blocks = [
                    (x, y),
                    (x + 1, y),
                    (x, y + 1),
                    (x + 1, y + 1),
                ];

                let in_blocks = in_blocks.map(|uv| get_block_zero_borders(&self.front, uv));

                let (out_block, _) = self.kernel.exec(in_blocks);

                self.back[(i as usize, j as usize)] = out_block;
            }
        }

        std::mem::swap(&mut self.back, &mut self.front);
        self.zero_borders = !self.zero_borders;
    }

    /// Returns (width, height) in pixels
    pub fn pixel_dims(&self) -> (usize, usize) {
        let w = calc_block_width(&*self.kernel);
        ((self.front.width() - 1) * w, (self.front.height() - 1) * w)
    }

    fn index_block_pixel(&self, index: (usize, usize)) -> ((usize, usize), (usize, usize)) {
        let (mut x, mut y) = index;

        let w = calc_block_width(&*self.kernel);

        // Every other frame, the blocks are in an alternate configuration where the result of the
        // last frame is offset by half a block width
        if !self.zero_borders {
            x += w / 2;
            y += w / 2;
        }

        ((x / w, y / w), (x % w, y % w))
    }

    pub fn get_pixel(&self, index: (usize, usize)) -> bool {
        let (block_idx, pixel_idx) = self.index_block_pixel(index);
        //dbg!(index, block_idx, pixel_idx);
        //dbg!();
        self.front[block_idx][pixel_idx]
    }

    pub fn set_pixel(&mut self, index: (usize, usize), val: bool) {
        let (block_idx, pixel_idx) = self.index_block_pixel(index);
        self.front[block_idx][pixel_idx] = val;
    }

    pub fn data_mut(&mut self) -> &mut Array2D<Block> {
        &mut self.front
    }
}

fn get_block_zero_borders(arr: &Array2D<Block>, xy: (i32, i32)) -> Block {
    let (x, y) = xy;
    if x < 0 || y < 0 || x >= arr.width() as i32 || y >= arr.height() as i32 {
        let mut out = arr[(0, 0)].clone();
        out.data_mut().iter_mut().for_each(|x| *x = false);
        out
    } else {
        arr[(x as usize, y as usize)].clone()
    }
}

pub fn calc_block_width(ker: &dyn Kernel) -> usize {
    1 << ker.order()
}
