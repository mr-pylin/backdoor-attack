import numpy as np
from enum import Enum

class PatternType(Enum):
    SOLID                 = 'solid'
    CHECKERBOARD          = 'checkerboard'
    GAUSSIAN              = 'gaussian'
    LEAST_SIGNIFICANT_BIT = 'least_significant_bit'

class Pattern:
    def __init__(self, dtype: type, shape: tuple[int]) -> None:
        """
        Create & Apply patterns to the dataset.

        Args:
            - `dtype` (type): dtype of the dataset e.g. np.uint8
            - `shape` (tuple[int]): shape of the dataset e.g. (28, 28, 1) or (28, 28)
        
        Returns:
            None
        """
        self.dtype  = dtype
        self.height = shape[0]
        self.width  = shape[1]
        self.depth  = shape[2] if len(shape) == 3 else 1

    def apply(self, subset: np.ndarray, pattern_type: PatternType, **kwargs) -> tuple[np.ndarray]:
        """
        Apply a pattern to the dataset.

        Args:
            - `subset` (np.ndarray): 
            - `pattern_type` (PatternType): Type of pattern to apply
            - kwargs: Additional arguments based on the pattern type
            
        Returns:
            tuple[np.ndarray]: (clean_set, poison_set), pattern
        """

        clean_set  = subset.copy()

        # generate the pattern
        pattern = self._generate_pattern(pattern_type, **kwargs)

        # apply the pattern to the subset
        poison_set = self._apply_pattern(subset, pattern_type, pattern, **kwargs)

        return (clean_set, poison_set), pattern



    def _generate_pattern(self, pattern_type: PatternType, **kwargs) -> np.ndarray:
        """
        Generate a pattern based on the type.

        Args:
            - `pattern_type` (PatternType): Type of pattern to generate
            - kwargs: Additional arguments based on the pattern type
        
        Returns:
            - np.ndarray
        """
        if pattern_type == PatternType.SOLID:
            return self._solid_pattern(**kwargs)
        elif pattern_type == PatternType.CHECKERBOARD:
            return self._checkerboard_pattern(**kwargs)
        elif pattern_type == PatternType.GAUSSIAN:
            return self._gaussian_noise_pattern(**kwargs)
        elif pattern_type == PatternType.LEAST_SIGNIFICANT_BIT:
            return self._least_significant_bits_pattern(**kwargs)
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")
        


    def _apply_pattern(self, subset: np.ndarray, pattern_type: PatternType, pattern: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply a pattern to the dataset.

        Args:
            - `subset` (np.ndarray): 
            - `pattern` (np.ndarray): Pattern to apply
        
        Returns:
            - np.ndarray
        """

        if pattern_type == PatternType.SOLID:
            pattern_height_pos = slice(0 + kwargs['pattern_pos'][0], pattern.shape[0] + kwargs['pattern_pos'][0])
            pattern_width_pos  = slice(0 + kwargs['pattern_pos'][1], pattern.shape[1] + kwargs['pattern_pos'][1])
            subset[:, pattern_height_pos, pattern_width_pos] = pattern[None, :, :]

        elif pattern_type == PatternType.CHECKERBOARD:
            pattern_height_pos = slice(0 + kwargs['pattern_pos'][0], pattern.shape[0] + kwargs['pattern_pos'][0])
            pattern_width_pos  = slice(0 + kwargs['pattern_pos'][1], pattern.shape[1] + kwargs['pattern_pos'][1])
            subset[:, pattern_height_pos, pattern_width_pos] = pattern[None, :, :]

        elif pattern_type == PatternType.GAUSSIAN:
            subset = subset.astype(np.float64)
            subset += pattern
            subset = subset.clip(min= np.iinfo(self.dtype).min, max= np.iinfo(self.dtype).max)
            subset = subset.astype(self.dtype)
            
        elif pattern_type == PatternType.LEAST_SIGNIFICANT_BIT:
            subset &= pattern

        return subset

    def _solid_pattern(self, **kwargs) -> np.ndarray:
        """
        Create a solid pattern.

        Args:
            - `pattern_size` (tuple[int]): Size of the pattern. must be less than image width & height.
                - e.g. `(3, 4)` is a pattern with height= 3 and width= 4.
            - `fill_value` (int, optional): A value to fill the pattern in the range `[0, 2**L]` where L is the BPP(bits per pixel) of the dataset.
                - Defaults to 255.
            - `shape` (str, optional).
                - possible values: {'rectangle', 'ellipsis'}
                - Defaults to 'rectangle'.

        Returns:
            - np.ndarray
        """

        # default parameters value
        defaults = {
            'fill_value': 255,
            'shape'     : 'rectangle',
        }
        kwargs = {**defaults, **kwargs}

        # check `size` to be smaller than the size of the images in the dataset
        assert kwargs['pattern_size'][0] < self.height and kwargs['pattern_size'][1] < self.width, f"Invalid `size` value: {kwargs['pattern_size']}; it should be less than image size which is {self.height, self.width}"

        # check `fill_value` to be compatible with the dataset dtype
        assert np.iinfo(self.dtype).min <= kwargs['fill_value'] <= np.iinfo(self.dtype).max, f"Invalid `fill_value` value: {kwargs['fill_value']}; should be in range {[np.iinfo(self.dtype).min, np.iinfo(self.dtype).max]}."

        if kwargs['shape'] == 'rectangle':
            solid_pattern = np.full(shape= kwargs['pattern_size'], fill_value= kwargs['fill_value'], dtype= self.dtype)
        
        elif kwargs['shape'] == 'ellipsis':
            a, b = (kwargs['pattern_size'][0] - 1) / 2, (kwargs['pattern_size'][1] - 1) / 2

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
            
            solid_pattern[mask] = kwargs['fill_value']

            solid_pattern = solid_pattern.astype(self.dtype)
        
        else:
            raise ValueError(f"Invalid `shape` value: {kwargs['shape']}; should be {{'rectangle', 'ellipsis'}}.")
    

        # repeat by depth
        solid_pattern = solid_pattern[:, :, None]
        solid_pattern = np.repeat(solid_pattern, self.depth, axis= 2)

        return solid_pattern
        

    def _checkerboard_pattern(self, **kwargs) -> np.ndarray:
        """
        Create a checkerboard pattern.

        Args:
            - `pattern_size` (tuple[int]): Size of the pattern. must be less than image width & height.
                - e.g. `(3, 4)` is a pattern with height= 3 and width= 4.
            - `fill_value` (int, optional): A value to fill the pattern in the range `[0, 2**L]` where L is the BPP(bits per pixel) of the dataset.
                - Defaults to 255.
            - `compliment` (bool, optional): Compliments the positions
                - possible values: {True, False}
                - Defaults to False.
        
        Returns:
            - np.ndarray
        """

        # default parameters value
        defaults = {
            'fill_value': 255,
            'compliment': False,
        }
        kwargs = {**defaults, **kwargs}

        # check `size` to be smaller than the size of the images in the dataset
        assert kwargs['pattern_size'][0] < self.height and kwargs['pattern_size'][1] < self.width, f"Invalid `size` value: {kwargs['pattern_size']}; it should be less than image size which is {self.height, self.width}"

        # check `fill_value` to be compatible with the dataset dtype
        assert np.iinfo(self.dtype).min <= kwargs['fill_value'] <= np.iinfo(self.dtype).max, f"Invalid `fill_value` value: {kwargs['fill_value']}; should be in range {[np.iinfo(self.dtype).min, np.iinfo(self.dtype).max]}."
        
        # create checkerboard pattern with 0-1 values
        checkerboard_pattern = (np.indices(dimensions= kwargs['pattern_size']).sum(axis = 0) % 2).astype(self.dtype)

        if kwargs['compliment']:
            checkerboard_pattern = 1 - checkerboard_pattern
        
        # fill values
        checkerboard_pattern *= kwargs['fill_value']

        # repeat by depth
        checkerboard_pattern = checkerboard_pattern[:, :, None]
        checkerboard_pattern = np.repeat(checkerboard_pattern, self.depth, axis= 2)

        return checkerboard_pattern


    def _gaussian_noise_pattern(self, **kwargs) -> np.ndarray:
        """
        Create a gaussian_noise pattern.

        Args:
            - `mu` (int, optional): The Mean of the gaussian noise distribution
                - Defaults to 0.
            - `std` (int, optional): The Standard Deviation of the gaussian noise distribution
                - Defaults to 5.
            - `seed` (int): set a fixed seed to get the same gaussian
                - Defaults t0 42.
        
        Returns:
            - np.ndarray
        """

        # default parameters value
        defaults = {
            'mu'  : 0,
            'std' : 5,
            'seed': 42,
        }
        kwargs = {**defaults, **kwargs}

        # set a fixed seed
        np.random.seed(kwargs['seed'])
        
        gaussian_pattern = np.random.normal(loc= kwargs['mu'], scale= kwargs['std'], size= (self.height, self.width, self.depth))

        return gaussian_pattern


    def _least_significant_bits_pattern(self, **kwargs) -> np.ndarray:
        """
        Create a least_significant_bits pattern.

        Args:
            - `mask` (str): sequence of 0 and 1 followed by number of image bits e.g. "11111100"
        
        Returns:
            - np.ndarray
        """

        # check `mask` length matches with the number of bits per pixel of images
        assert len(kwargs['mask']) == np.iinfo(self.dtype).bits, f"Invalid `mask` value: {kwargs['mask']}; the length of the mask should be equal to number of bits per pixel of images which is {np.iinfo(self.dtype).bits}"

        # check `mask` to be a string of values {'0', '1'}
        assert set(kwargs['mask']).issubset({'0', '1'}), f"Invalid `mask` value: {kwargs['mask']}; only a sequence of 0 and 1 is allowed"

        mask_pattern = int(kwargs['mask'], 2)

        return mask_pattern

if __name__ == '__main__':

    # dependencies
    from torchvision.datasets import CIFAR10
    import matplotlib.pyplot as plt

    # load a dataset
    dataset = CIFAR10(root= '../datasets/', train= True, transform= None, download= False)

    # create a subset
    batch_size = 2
    subset = dataset.data[:batch_size]

    # initialize the Pattern class
    pattern_generator = Pattern(dtype= subset.dtype, shape= subset[0].shape)

    # apply a solid pattern
    pattern_params_1 = {'pattern_size': (3, 3), 'pattern_pos': (28, 28), 'fill_value': 255, 'shape': 'rectangle'}
    (clean_set_1, poison_set_1), solid_pattern_1 = pattern_generator.apply(subset.copy(), PatternType.SOLID, **pattern_params_1)
    print("solid pattern_1:")
    print(solid_pattern_1)

    pattern_params_2 = {'pattern_size': (3, 3), 'pattern_pos': (28, 28), 'fill_value': 255, 'shape': 'ellipsis'}
    (clean_set_2, poison_set_2), solid_pattern_2 = pattern_generator.apply(subset.copy(), PatternType.SOLID, **pattern_params_2)
    print("solid pattern_2:")
    print(solid_pattern_2)

    # apply a checkerboard pattern
    pattern_params_3 = {'pattern_size': (3, 3), 'pattern_pos': (28, 28), 'fill_value': 255, 'compliment': True}
    (clean_set_3, poison_set_3), checkerboard_pattern = pattern_generator.apply(subset.copy(), PatternType.CHECKERBOARD, **pattern_params_3)
    print("checkerboard pattern:")
    print(checkerboard_pattern)

    # apply a gaussian pattern
    pattern_params_4 = {'mu': 0, 'std': 10, 'seed': 42}
    (clean_set_4, poison_set_4), gaussian_pattern = pattern_generator.apply(subset.copy(), PatternType.GAUSSIAN, **pattern_params_4)
    print("gaussian pattern:")
    print(gaussian_pattern.shape)

    # apply a least significant bit pattern
    pattern_params_5 = {'mask': '11110000'}
    (clean_set_5, poison_set_5), least_significant_bit_pattern = pattern_generator.apply(subset.copy(), PatternType.LEAST_SIGNIFICANT_BIT, **pattern_params_5)
    print("least_significant bit pattern pattern:")
    print(least_significant_bit_pattern)

    # plot
    fig, axs = plt.subplots(nrows= 3, ncols= 5, figsize= (16, 12), layout= 'compressed')

    axs[0, 0].imshow(solid_pattern_1)
    axs[0, 0].set(title= 'solid_pattern_1', xticks= range(pattern_params_1['pattern_size'][0]), yticks= range(pattern_params_1['pattern_size'][1]))
    axs[0, 1].imshow(solid_pattern_2)
    axs[0, 1].set(title= 'solid_pattern_2', xticks= range(pattern_params_2['pattern_size'][0]), yticks= range(pattern_params_2['pattern_size'][1]))
    axs[0, 2].imshow(checkerboard_pattern)
    axs[0, 2].set(title= 'checkerboard', xticks= range(pattern_params_3['pattern_size'][0]), yticks= range(pattern_params_3['pattern_size'][1]))
    axs[0, 3].imshow(gaussian_pattern)
    axs[0, 3].set(title= 'gaussian', xticks= range(0, subset.shape[1], int(subset.shape[1] // 4)), yticks= range(0, subset.shape[2], int(subset.shape[2] // 4)))
    # axs[0, 4].imshow(least_significant_bit_pattern)
    axs[0, 4].set(title= f"lsb pattern: {bin(least_significant_bit_pattern)[2:]}")

    axs[1, 0].imshow(poison_set_1[0], vmin= 0, vmax= 255)
    axs[1, 0].set_title('sample 1')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(poison_set_2[0], vmin= 0, vmax= 255)
    axs[1, 1].set_title('sample 1')
    axs[1, 1].axis('off')
    axs[1, 2].imshow(poison_set_3[0], vmin= 0, vmax= 255)
    axs[1, 2].set_title('sample 1')
    axs[1, 2].axis('off')
    axs[1, 3].imshow(poison_set_4[0], vmin= 0, vmax= 255)
    axs[1, 3].set_title('sample 1')
    axs[1, 3].axis('off')
    axs[1, 4].imshow(poison_set_5[0], vmin= 0, vmax= 255)
    axs[1, 4].set_title('sample 1')
    axs[1, 4].axis('off')

    axs[2, 0].imshow(poison_set_1[1], vmin= 0, vmax= 255)
    axs[2, 0].set_title('sample 2')
    axs[2, 0].axis('off')
    axs[2, 1].imshow(poison_set_2[1], vmin= 0, vmax= 255)
    axs[2, 1].set_title('sample 2')
    axs[2, 1].axis('off')
    axs[2, 2].imshow(poison_set_3[1], vmin= 0, vmax= 255)
    axs[2, 2].set_title('sample 2')
    axs[2, 2].axis('off')
    axs[2, 3].imshow(poison_set_4[1], vmin= 0, vmax= 255)
    axs[2, 3].set_title('sample 2')
    axs[2, 3].axis('off')
    axs[2, 4].imshow(poison_set_5[1], vmin= 0, vmax= 255)
    axs[2, 4].set_title('sample 2')
    axs[2, 4].axis('off')

    plt.show()