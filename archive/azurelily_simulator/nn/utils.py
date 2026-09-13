def divide_up(a, b):
    return (a + b - 1) // b

def next_power_of_2(a):
    return 1 if a == 0 else 2**(a - 1).bit_length()
