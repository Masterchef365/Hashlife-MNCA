use crate::array2d::Array2D;

/// Block data, whose size is known by the Kernel
#[derive(Clone, Debug, Default)]
pub struct Block(Box<[bool]>);

pub trait Kernel {
    /// Power law size of the basic block. E.g. each block has a width of 2^n, where n = self.order()
    /// For HashLife/Conway life (radius 1), this would be 2, since input would be 4 2x2 blocks and output would be 1 2x2 block
    /// For an MNCA with radius 4, this would be 3, since the input would be 4 8x8 blocks and output would be 1 8x8 block
    fn order(&self) -> usize;

    /// Given a novel combination of 4 blocks, produce an output block advanced by one time step
    /// (each entry in the input and output blocks are either 0 or 1 indicating dead or live states respectively)
    fn approximate(&mut self, blocks: [Block; 4]) -> (Block, KernelResult);
}

#[derive(Clone, Debug, Copy)]
pub enum KernelResult {
    /// An exact, new block has been returned. This new block should be hashed and inserted into the macrocell tree.
    NewBlock,
    /// An approximate block has been returned. This new block should be hashed and should be found to already be in the macrocell tree.
    Approximate,
}

struct Dense {
    back: Array2D<Block>,
    front: Array2D<Block>,
    kernel: Box<dyn Kernel>,
    zero_borders: bool,
}

impl Dense {
    pub fn new(kernel: Box<dyn Kernel>, width: usize, height: usize) -> Self {
        // To account for difference in size between frames, we add 1 to width and height
        let zeros = vec![Block::zero(&*kernel); (width + 1) * (height + 1)];

        Self {
            front: Array2D::from_array(width, zeros.clone()),
            back: Array2D::from_array(width, zeros),
            kernel,
            zero_borders: true,
        }
    }

    pub fn step(&mut self) {
        for i in 0..(self.front.width() - 1) as i32 {
            for j in 0..(self.front.height() - 1) as i32 {
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

                let (out_block, _) = self.kernel.approximate(in_blocks);

                self.back[(i as usize, j as usize)] = out_block;
            }
        }

        std::mem::swap(&mut self.back, &mut self.front);
        self.zero_borders = !self.zero_borders;
    }

    /// Returns (width, height) in pixels
    pub fn pixel_dims(&self) -> (usize, usize) {
        let w = Block::width(&*self.kernel);
        (self.front.width() * w, self.front.height() * w)
    }

    fn index_block_pixel(&self, index: (i32, i32)) -> (usize, usize) {
        let (mut x, mut y) = index;

        let w = Block::width(&*self.kernel);

        // Every other frame, the blocks are in an alternate configuration where the result of the
        // last frame is offset by half a block width
        if !self.zero_borders {
            x += w as i32 / 2;
            y += w as i32 / 2;
        }

        (x as usize, y as usize)
    }

    pub fn get_pixel(&self, index: (i32, i32)) -> bool {
        let block_idx = self.index_block_pixel(index);
        self.front[block_idx].get(&*self.kernel, index)
    }

    pub fn set_pixel(&mut self, index: (i32, i32), val: bool) {
        let block_idx = self.index_block_pixel(index);
        self.front[block_idx].set(&*self.kernel, index, val);
    }
}

fn get_block_zero_borders(arr: &Array2D<Block>, xy: (i32, i32)) -> Block {
    let (x, y) = xy;
    if x < 0 || y < 0 || x >= arr.width() as i32 || y >= arr.height() as i32 {
        Block::zeros_like(&arr[(0, 0)])
    } else {
        arr[(x as usize, y as usize)].clone()
    }
}

impl Block {
    pub fn zero(ker: &dyn Kernel) -> Self {
        Self::new(ker, vec![false; Self::width(ker).pow(2)])
    }

    pub fn new(ker: &dyn Kernel, data: Vec<bool>) -> Self {
        let expected_len = Self::width(ker).pow(2);
        assert_eq!(expected_len, data.len());
        Self(data.into_boxed_slice())
    }

    pub fn width(ker: &dyn Kernel) -> usize {
        1 << ker.order()
    }

    pub fn data_mut(Block(data): &mut Self) -> &mut [bool] {
        data
    }

    pub fn data(Block(data): &Self) -> &[bool] {
        data
    }

    pub fn zeros_like(other: &Self) -> Self {
        Self(vec![false; other.0.len()].into_boxed_slice())
    }

    /// Get a subpixel within a block by x and y coordinates
    /// X and Y wrap around block size, which is in units of 2^n
    pub fn index(ker: &dyn Kernel, xy: (i32, i32)) -> usize {
        let block_width = Self::width(ker) as i32;
        let (x, y) = xy;
        let (x, y) = (x % block_width, y % block_width);
        (x + block_width * y) as usize
    }

    pub fn get(&self, ker: &dyn Kernel, xy: (i32, i32)) -> bool {
        let Self(data) = self;
        data[Self::index(ker, xy)]
    }

    pub fn set(&mut self, ker: &dyn Kernel, xy: (i32, i32), val: bool) {
        let Self(data) = self;
        data[Self::index(ker, xy)] = val;
    }
}
