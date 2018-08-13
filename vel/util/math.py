

def divide_ceiling(numerator, denominator):
    """ Determine the smallest number k such, that denominator * k >= numerator """
    split_val = numerator // denominator
    rest = numerator % denominator

    if rest > 0:
        return split_val + 1
    else:
        return split_val
