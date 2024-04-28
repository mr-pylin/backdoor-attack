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

    def apply(self, subset: np.ndarray, pattern_type: str, pattern_size: tuple[int], pattern_pos: tuple[int], fill_value: int = 255, **kwargs) -> tuple[np.ndarray]:
        """
        Args:
            - `subset` (np.ndarray): 
            - `pattern_type` (type): 
            - `pattern_size` (type): 
            - `pattern_pos` (type): 
            - `fill_value` (type): 
            - kwargs :
                - if pattern_type == 'solid':
                    - `shape`: (str, optional). possible values = {'rectangle', 'ellipsis'}. Defaults to 'rectangle'.
                - if pattern_type == 'checkerboard':
                    - `compliment`: (bool, optional). possible values = {True, False}. Defaults to False.
        
        Returns tuple[np.ndarray]: (clean_set, poison_set), pattern
        """

        # create the pattern
        if pattern_type == 'solid':
            shape = kwargs['shape']
            if shape:
                pattern = self.__solid_pattern(pattern_size, fill_value, shape)
            else:
                pattern = self.__solid_pattern(pattern_size, fill_value)

        elif pattern_type == 'checkerboard':
            compliment = kwargs.get('compliment')
            if compliment:
                pattern = self.__checkerboard_pattern(pattern_size, fill_value, compliment)
            else:
                pattern = self.__checkerboard_pattern(pattern_size, fill_value)
            
        else:
            raise ValueError(f"Invalid `pattern_type` value: {pattern_type}; should be {{'solid', 'checkerboard'}}.")
        
        # apply the pattern
        clean_set = subset.copy()

        pattern_height_pos = slice(pattern_pos[0], pattern_pos[0] + pattern.shape[0])
        pattern_width_pos  = slice(pattern_pos[1], pattern_pos[1] + pattern.shape[1])

        subset[:, pattern_height_pos, pattern_width_pos] = pattern[None, :, :, None]

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
            return solid_pattern
        
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

            return solid_pattern.astype(self.dtype)
        
        else:
            raise ValueError(f"Invalid `shape` value: {shape}; should be {{'rectangle', 'ellipsis'}}.")
        

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

        return checkerboard_pattern


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
    (_, poison), checkerboard_pattern_2 = pattern.apply(subset, 'checkerboard', pattern_size= (3, 3), pattern_pos= (28, 28), fill_value= 255, compliment= True)
    print(checkerboard_pattern_1)
    print(checkerboard_pattern_2)

    # plot
    fig, axs = plt.subplots(nrows= 1, ncols= 1, figsize= (4, 4), layout= 'compressed')

    axs.imshow(poison[0])
    axs.axis('off')

    plt.show()
