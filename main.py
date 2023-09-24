# Фио: Яшин Никита Андреевич
# Почта: niknlo007@gmail.com
# Телефон: +79163679923
# Телеграм: @AnonimYYYs


import matplotlib.pyplot as plt


# коэффициенты для f и сама функция
k0, k1, k2, k3, k4 = -243, -405, -270, -90, -15

def f(y0, y1, y2, y3, y4):
    return k0 * y0 + k1 * y1 + k2 * y2 + k3 * y3 + k4 * y4


def runge_kutta_4th_order(x0, y0, yp0, ypp0, yppp0, ypppp0, x_final, h):
    # Инициализация списков для хранения результатов
    x_values = []
    y_values = []

    # Начальные условия
    x = x0
    y = y0
    yp = yp0
    ypp = ypp0
    yppp = yppp0
    ypppp = ypppp0

    while x < x_final:
        x_values.append(x)
        y_values.append(y)

        # Вычисление коэффициентов шага метода Рунге-Кутты
        k1 = h * yp
        l1 = h * ypp
        m1 = h * yppp
        n1 = h * ypppp

        k2 = h * (yp + 0.5 * l1)
        l2 = h * (ypp + 0.5 * m1)
        m2 = h * (yppp + 0.5 * n1)
        n2 = h * f(y + 0.5 * k1, yp + 0.5 * l1, ypp + 0.5 * m1, yppp + 0.5 * n1, ypppp + 0.5 * n1)

        k3 = h * (yp + 0.5 * l2)
        l3 = h * (ypp + 0.5 * m2)
        m3 = h * (yppp + 0.5 * n2)
        n3 = h * f(y + 0.5 * k2, yp + 0.5 * l2, ypp + 0.5 * m2, yppp + 0.5 * n2, ypppp + 0.5 * n2)

        k4 = h * (yp + l3)
        l4 = h * (ypp + m3)
        m4 = h * (yppp + n3)
        n4 = h * f(y + k3, yp + l3, ypp + m3, yppp + n3, ypppp + n3)

        # Обновление значений
        x += h
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yp += (l1 + 2 * l2 + 2 * l3 + l4) / 6
        ypp += (m1 + 2 * m2 + 2 * m3 + m4) / 6
        yppp += (n1 + 2 * n2 + 2 * n3 + n4) / 6

    x_values.append(x)
    y_values.append(y)

    return x_values, y_values




y00, y10, y20, y30, y40 = 0, 3, -9, -8, 0
x0, x_max = 0, 5
h = 0.01

x_values, y_values = runge_kutta_4th_order(x0, y00, y10, y20, y30, y40, x_max, h)



# Вывод результата
plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Численное решение уравнения Коши')
plt.grid(True)
plt.show()



h1, h2, h3, h4 = 0.01, 0.001, 0.0001, 0.00001

x1_values, y1_values = runge_kutta_4th_order(x0, y00, y10, y20, y30, y40, x_max, h1)
x2_values, y2_values = runge_kutta_4th_order(x0, y00, y10, y20, y30, y40, x_max, h2)
x3_values, y3_values = runge_kutta_4th_order(x0, y00, y10, y20, y30, y40, x_max, h3)
x4_values, y4_values = runge_kutta_4th_order(x0, y00, y10, y20, y30, y40, x_max, h4)


# Вывод результата
plt.figure(figsize=(10, 5))
plt.plot(x1_values, y1_values, label=f'h={h1}', color='k')
plt.plot(x2_values, y2_values, label=f'h={h2}', color='r')
plt.plot(x3_values, y3_values, label=f'h={h3}', color='g')
plt.plot(x4_values, y4_values, label=f'h={h4}', color='b')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Сходимость численного решения уравнения Коши')
plt.grid(True)
plt.legend()
plt.show()

