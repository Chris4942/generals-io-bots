from mapper import convert_moves_to_machine_output

x, y, direction = convert_moves_to_machine_output({
    "start": 178,
    "end": 163,
})

print(f'{x} {y} {direction}')