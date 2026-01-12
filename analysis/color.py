import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

plt.rcParams['font.family'] = 'Meiryo'
# パレット
PALETTE = {
    "adgreen":["#01916D","#34A78A","#67BDA7","#99D3C5","#CCE9E2","#E6F4F0","#014937"],
    "gray":["#333333","#5C5C5C","#858585","#ADADAD","#D6D6D6","#EBEBEB"],
    "blue":["#1E83BE","#4B9CCB","#78B5D8","#A5CDE5","#D2E6F2","#0F425F",],
    "purple":["#8E58AD","#A579BD","#BB9BCE","#D2BCDE","#E8DEEF","#F4EEF7","#472C57"],
    "pink":["#E0356C","#E65D89","#EC86A7","#F9D7E2","#FCEBF0","#701B36"],
    "orange":["#EA5504","#EE7736","#F29968","#F7BB9B","#FBDDCD","#FDEEE6","#752B02"],
    "lightgreen":["#67AC1E","#85BD4B","#A4CD78","#C2DEA5","#E1EED2","#F0F7E9","#34560F"],
    "red":["#FB0020"],
    "black":["#000000"]
}
PALETTE_np = np.array(PALETTE)
print(PALETTE["blue"][0])

x = np.linspace(0, 10, 100)
plt.figure(figsize=(10, 6))
for i, (family, colors) in enumerate(PALETTE.items()):
    for level, color in enumerate(colors, start=1):
        y = np.sin(x) + i  + level * 0.15
        plt.plot(x, y, color=color)

plt.title("色確認デモ")
plt.xlabel("x")
plt.ylabel("y (offset)")
plt.show()
