import matplotlib.pyplot as plt

# ТВОЇ ДАНІ:
# Кількість воркерів
workers = [1, 2, 4, 8, 16] # Заміни 16 на свій MAX_CORES, якщо він інший
# Встав сюди Wall time, які ти виміряв для процесів:
wall_times = [10.0, 5.5, 3.2, 2.1, 1.8] # ПРИКЛАД! Встав свої реальні числа!

# Обчислення прискорення (Speedup = T1 / Tn)
t1 = wall_times[0]
speedups = [t1 / t for t in wall_times]

plt.figure(figsize=(8, 5))
# Реальне прискорення
plt.plot(workers, speedups, marker='o', linestyle='-', color='b', label='Реальне прискорення (Processes)')
# Ідеальне лінійне прискорення
plt.plot(workers, workers, linestyle='--', color='r', label='Ідеальне прискорення')

plt.title('Графік прискорення індексації (Закон Амдала)')
plt.xlabel('Кількість воркерів')
plt.ylabel('Прискорення (Speedup)')
plt.xticks(workers)
plt.legend()
plt.grid(True)

plt.savefig('acceleration.png')
print("Графік успішно збережено як acceleration.png")