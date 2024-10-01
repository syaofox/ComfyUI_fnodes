def make_even(number):
    _, remainder = divmod(number, 2)
    return number + remainder
