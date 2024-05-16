import numpy as np

class Pattern:
    def __init__(self, dtype: type, shape: tuple[int]) -> None:
        """
        Create & Apply patterns to the dataset.

        Args:
            - `dtype` (type): dtype of the dataset e.g. np.uint8
            - `shape` (tuple[int]): shape of the dataset e.g. (28, 28, 1) or (28, 28)
        
        Returns:
            - None
        """
        self.dtype  = dtype
        self.height = shape[0]
        self.width  = shape[1]
        self.depth  = shape[2] if len(shape) == 3 else 1

    def apply(self, subset: np.ndarray, pattern_type: str, **kwargs) -> tuple[np.ndarray]:
        """
        Args:
            - `subset` (np.ndarray): 
            - `pattern_type` (str): 
                - possible values = {'solid', 'checkerboard', 'gaussian'}
            - kwargs :
                - if pattern_type == 'solid':
                    - `pattern_size` (tuple[int]): 
                    - `pattern_pos` (tuple[int]): 
                    - `fill_value` (int): [Defaults to 255]
                    - `shape`: (str, optional). possible values = {'rectangle', 'ellipsis'}. Defaults to 'rectangle'.
                - if pattern_type == 'checkerboard':
                    - `pattern_size` (tuple[int]): 
                    - `pattern_pos` (tuple[int]): 
                    - `fill_value` (int): [Defaults to 255]
                    - `compliment`: (bool, optional). possible values = {True, False}. Defaults to False.
                - if pattern_type == 'gaussian':
                    - `mu` (int): The Mean of the gaussian noise distribution [Defaults to 0]
                    - `std` (int): The Standard Deviation of the gaussian noise distribution [Defaults to 1]
                    - `seed` (int): set a fixed seed to get the same gaussian [Defaults to 42]
        
        Returns tuple[np.ndarray]: (clean_set, poison_set), pattern
        """

        clean_set = subset.copy()

        # create the pattern
        if pattern_type == 'solid':

            pattern_size = kwargs.get('pattern_size')
            pattern_pos  = kwargs.get('pattern_pos')
            fill_value   = kwargs.get('fill_value')
            shape        = kwargs.get('shape')

            if shape:
                pattern = self.__solid_pattern(pattern_size, fill_value, shape)
            else:
                pattern = self.__solid_pattern(pattern_size, fill_value)

            # apply the pattern
            pattern_height_pos = slice(pattern_pos[0], pattern_pos[0] + pattern.shape[0])
            pattern_width_pos  = slice(pattern_pos[1], pattern_pos[1] + pattern.shape[1])
            subset[:, pattern_height_pos, pattern_width_pos] = pattern[None, :, :]

        elif pattern_type == 'checkerboard':

            pattern_size = kwargs.get('pattern_size')
            pattern_pos  = kwargs.get('pattern_pos')
            fill_value   = kwargs.get('fill_value')
            compliment   = kwargs.get('compliment')

            if not fill_value:
                fill_value = 255

            if compliment:
                pattern = self.__checkerboard_pattern(pattern_size, fill_value, compliment)
            else:
                pattern = self.__checkerboard_pattern(pattern_size, fill_value)

            # apply the pattern
            pattern_height_pos = slice(pattern_pos[0], pattern_pos[0] + pattern.shape[0])
            pattern_width_pos  = slice(pattern_pos[1], pattern_pos[1] + pattern.shape[1])
            subset[:, pattern_height_pos, pattern_width_pos] = pattern[None, :, :]
        
        elif pattern_type == 'gaussian':

            mu   = kwargs.get('mu')
            std  = kwargs.get('std')
            seed = kwargs.get('seed')
            pattern_pos = (0, 0)

            if not mu  : mu = 0
            if not std : std = 1
            if not seed: seed == 42

            # set the seed
            np.random.seed(seed)

            pattern = self.__gaussian_noise_pattern(mu, std)

            # apply the pattern
            pattern_height_pos = slice(pattern_pos[0], pattern_pos[0] + pattern.shape[0])
            pattern_width_pos  = slice(pattern_pos[1], pattern_pos[1] + pattern.shape[1])

            subset = subset.astype(np.float64)
            subset[:, pattern_height_pos, pattern_width_pos] += pattern[None, :, :]
            subset = subset.clip(min= np.iinfo(self.dtype).min, max= np.iinfo(self.dtype).max)
            subset = subset.astype(self.dtype)

        else:
            raise ValueError(f"Invalid `pattern_type` value: {pattern_type}; should be {{'solid', 'checkerboard', 'gaussian'}}.")

        return (clean_set, subset), pattern

    def __solid_pattern(self, size: tuple[int], fill_value: int = 255, shape: str = 'rectangle') -> np.ndarray:
        """
        Create a solid pattern.

        Args:
            - `size` (tuple[int]): Size of the pattern. must be less than image width & height.
                - e.g. `(3, 4)` is a pattern with height= 3 and width= 4.
            - `fill_value` (int, optional): A value to fill the pattern in the range `[0, 2**L]` where L is the BPP(bits per pixel) of the dataset.
                - Defaults to 255.
            - `shape` (str, optional).
                - possible values: {'rectangle', 'ellipsis'}
                - Defaults to 'rectangle'.

        Returns:
            - np.ndarray
        """

        # check `size` to be smaller than the size of the images in the dataset
        assert size[0] < self.height and size[1] < self.width, f"Invalid `size` value: {size}; it should be less than image size which is {self.height, self.width}"

        # check `fill_value` to be compatible with the dataset dtype
        assert np.iinfo(self.dtype).min <= fill_value <= np.iinfo(self.dtype).max, f"Invalid `fill_value` value: {fill_value}; should be in range {[np.iinfo(self.dtype).min, np.iinfo(self.dtype).max]}."

        if shape == 'rectangle':
            solid_pattern = np.full(shape= size, fill_value= fill_value, dtype= self.dtype)
        
        elif shape == 'ellipsis':
            a, b = (size[0] - 1) / 2, (size[1] - 1) / 2

            # create an array of zeros
            array_size_y = int(a * 2 + 1)
            array_size_x = int(b * 2 + 1)
            solid_pattern = np.zeros((array_size_y, array_size_x))

            # create indices for x and y
            y,x = np.ogrid[-a: a + 1, -b: b + 1]

            # create the shape
            if a == b:
                mask = x ** 2 + y ** 2 <= a ** 2
            else:
                mask = x ** 2 / b ** 2 + y ** 2 / a ** 2 <= 1
            
            solid_pattern[mask] = fill_value

            solid_pattern = solid_pattern.astype(self.dtype)
        
        else:
            raise ValueError(f"Invalid `shape` value: {shape}; should be {{'rectangle', 'ellipsis'}}.")
    

        # repeat by depth
        solid_pattern = solid_pattern[:, :, None]
        solid_pattern = np.repeat(solid_pattern, self.depth, axis= 2)

        return solid_pattern
        

    def __checkerboard_pattern(self, size: tuple[int], fill_value: int = 255, compliment: bool = False) -> np.ndarray:
        """
        Create a checkerboard pattern.

        Args:
            - `size` (tuple[int]): Size of the pattern. must be less than image width & height.
                - e.g. `(3, 4)` is a pattern with height= 3 and width= 4.
            - `fill_value` (int, optional): A value to fill the pattern in the range `[0, 2**L]` where L is the BPP(bits per pixel) of the dataset.
                - Defaults to 255.
            - `compliment` (bool, optional): Compliments the positions
                - possible values: {True, False}
                - Defaults to False.
        
        Returns:
            - np.ndarray
        """

        # check `size` to be smaller than the size of the images in the dataset
        assert size[0] < self.height and size[1] < self.width, f"Invalid `size` value: {size}; it should be less than image size which is {self.height, self.width}"

        # check `fill_value` to be compatible with the dataset dtype
        assert np.iinfo(self.dtype).min <= fill_value <= np.iinfo(self.dtype).max, f"Invalid `fill_value` value: {fill_value}; should be in range {[np.iinfo(self.dtype).min, np.iinfo(self.dtype).max]}."
        
        # create checkerboard pattern with 0-1 values
        checkerboard_pattern = (np.indices(dimensions= size).sum(axis = 0) % 2).astype(self.dtype)

        if compliment:
            checkerboard_pattern = 1 - checkerboard_pattern
        
        # fill values
        checkerboard_pattern *= fill_value

        # repeat by depth
        checkerboard_pattern = checkerboard_pattern[:, :, None]
        checkerboard_pattern = np.repeat(checkerboard_pattern, self.depth, axis= 2)

        return checkerboard_pattern


    def __gaussian_noise_pattern(self, mu: int = 0, std: int = 1, seed: int = 42) -> np.ndarray:
        """
        Create a gaussian_noise pattern.

        Args:
            - `mu` (int, optional): The Mean of the gaussian noise distribution
                - Defaults to 0.
            - `std` (int, optional): The Standard Deviation of the gaussian noise distribution
                - Defaults to 1.
            - `seed` (int): set a fixed seed to get the same gaussian
                - Defaults t0 42.
        
        Returns:
            - np.ndarray
        """
        
        # create checkerboard pattern with 0-1 values
        gaussian_pattern = np.random.normal(loc= mu, scale= std, size= (self.height, self.width, self.depth))

        return gaussian_pattern


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # dataset
    subset = np.zeros(shape= (64, 32, 32, 3), dtype= np.uint8)

    # pattern object
    pattern = Pattern(dtype= subset.dtype, shape= subset[0].shape)

    # solid pattern
    _, solid_pattern_1 = pattern.apply(subset, 'solid', pattern_size= (3, 3), pattern_pos= (28, 28), fill_value= 255, shape= 'ellipsis')
    _, solid_pattern_2 = pattern.apply(subset, 'solid', pattern_size= (3, 3), pattern_pos= (28, 28), fill_value= 255, shape= 'rectangle')
    print(solid_pattern_1)
    print(solid_pattern_2)

    # checkerboard pattern
    _, checkerboard_pattern_1 = pattern.apply(subset, 'checkerboard', pattern_size= (3, 3), pattern_pos= (28, 28), fill_value= 255)
    _, checkerboard_pattern_2 = pattern.apply(subset, 'checkerboard', pattern_size= (3, 3), pattern_pos= (28, 28), fill_value= 255, compliment= True)
    print(checkerboard_pattern_1)
    print(checkerboard_pattern_2)

    # gaussian pattern
    (_, poison), gaussian_pattern_1 = pattern.apply(subset, 'gaussian', mu= 0, std= 5, seed= 42)
    print(gaussian_pattern_1.min())
    print(gaussian_pattern_1.max())

    # plot
    fig, axs = plt.subplots(nrows= 1, ncols= 1, figsize= (4, 4), layout= 'compressed')

    axs.imshow(poison[0], vmin= 0, vmax= 255)
    axs.axis('off')

    plt.show()
