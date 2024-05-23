import os

# for world in [
#     # *['8x8{}'.format(l) for l in 'ACD'],
#     # *['16x16{}'.format(l) for l in 'ABCDE'],
#     *['MediumAmbiguity', 'LargeGraph', 'LargeAmbiguity'],
# ]:
#     for discount in [0.5, 0.9, 0.95]:
#         os.system(
#             f"python comparison_cli.py --in=gridworlds/large/{world}.tmx --out=results/{world}_{discount}.json --horizon=80 --step_cost=30 --discount={discount} --flip_goals=0"
#         )

# world = '8x8A'
# discount = 0.9
# os.system(
#     f"python comparison_cli.py --in=gridworlds/square_rect_based/{world}.tmx --out=results/{world}_{discount}.json --horizon=80 --step_cost=30 --discount={discount} --flip_goals=0"
# )

discount = 0.9
os.system(
    f"python comparison_cli.py --in=gridworlds/large/MediumAmbiguity.tmx --out=results/MediumAmbiguity_{discount}_flip.json --horizon=80 --step_cost=30 --discount={discount} --flip_goals=1"
)

