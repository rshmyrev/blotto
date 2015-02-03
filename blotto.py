#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
'''Симулятор игры полковника Блотто'''
import random
import csv


# Функция генерируют рандомные стратегии
def generate(count, generator, mu=11, sigma=3, n_min_cells=0, max_in_min_cell=4):
    pool = {}  # пул стратегий
    for gamer in range(count):
        field = [0] * 9  # поле боя
        n_cells = 0  # количество заполненных клеток
        total = 100  # всего 100 солдат
        for i in random.sample(range(9), 9):  # заполняем случайную клетку
            if n_cells == 8:  # если клетка последняя
                cell = total  # записываем оставшихся солдат
            elif total == 0:  # если солдат не осталось
                cell = 0  # записываем в клетку 0
            else:
                if generator == 'random':  # генерируем случайные числа
                    cell = random.randint(0, total)
                elif generator == 'norm':  # числа с равномерным распределением
                    cell = round(random.gauss(mu, sigma))
                    # Может быть сгенерировано число меньше 0 или больше, чем
                    # кол-во оставшихся солдат. Тогда генерируем ещё раз.
                    while not (0 <= cell <= total):
                        cell = round(random.gauss(mu, sigma))
                elif generator == 'max-min':  # макс-минная стратегия
                    # Если необходимо заполнить n клеток минимальными цифрами
                    # выбираем случайное число от 0 до "max_in_min_cell"
                    if n_cells < n_min_cells:
                        cell = random.randint(0, max_in_min_cell)
                    else:  # для остальных записываем среднее +- отклонение
                        mean = round(total / (9 - n_cells))
                        cell = random.randint(mean - 1, mean + 1)
                        # Может быть сгенерировано число меньше 0 или больше,
                        # чем кол-во оставшихся солдат.
                        while not (0 <= cell <= total):
                            cell = random.randint(mean - 1, mean + 1)
            field[i] = cell  # заполняем клетку поля
            n_cells += 1  # увеличиваем счетчик заполненных клеток
            total -= cell  # уменьшаем количество оставшихся солдат
        pool[gamer] = field  # добавляем поле в пул стратегий
    print('done generating pool')
    return pool


# Функция сравнивает 2 стратегии и высчитывает очки, набранные каждой
def compare(strategy1, strategy2):
    gamer1_point = 0  # очки первой стратегии (gamer1)
    gamer2_point = 0  # очки второй стратегии (gamer2)
    for i in range(9):  # для каждой клетки на поле
        if strategy1[i] > strategy2[i]:  # победа первой стратегии
            gamer1_point += 1
        elif strategy2[i] > strategy1[i]:  # победа второй стратегии
            gamer2_point += 1
        else:  # ничья
            gamer1_point += 0.5
            gamer2_point += 0.5
    return(gamer1_point, gamer2_point)


# Функция проводит турнир между стратегиями
def tournament(pool):
    pool_copy = pool.copy()  # из копии пула будем удалять сыгравшие стратегии
    standings = {}  # турнирная таблица с очками
    for gamer in pool:  # заполняем турнирную таблицу нулями
        standings[gamer] = 0

    total_gamer = 0  # количество стратегий, сыгравших все матчи
    for gamer1 in pool:
        # Чтобы понимать, что скрипт не завис, для каждой сотне стратегий
        # будем выводить счетчик на экран
        total_gamer += 1
        if total_gamer % 100 == 0:
            print(total_gamer)

        strategy1 = pool_copy.pop(gamer1)  # достаем стратегию из пула
        for gamer2 in pool_copy:  # проводим матч с каждой оставшейся
            strategy2 = pool_copy[gamer2]
            gamer1_point, gamer2_point = compare(strategy1, strategy2)
            # добавляем очки в турнирную таблицу
            standings[gamer1] += gamer1_point
            standings[gamer2] += gamer2_point
    print('done tournament!')
    return(standings)


# Функция сортирует турнирную таблицу по убыванию количества очков и
# записывает в файл
def save_result(file_name, standings, pool):
    file_w = open('%s' % file_name, 'w', encoding='utf-8')
    gamers_list = sorted(standings.items(), key=lambda x: (x[1] * (-1), x[0]))
    for gamer, points in gamers_list:
        s = str("{0}\t{1}\t{2}\n").format(points, gamer, pool[gamer])
        file_w.write(s)
    file_w.close()


# Функция создает новую выборку стратегий исходя из рангов
def selection(standings, pool, ranks):
    select = {}
    gamers_list = sorted(standings.items(), key=lambda x: (x[1] * (-1), x[0]))
    for i in ranks:
        gamer = gamers_list[i][0]
        strategy = pool[gamer]
        select[gamer] = strategy
    return(select)


# Функция загружает ранее проведенный турнир
def load_result(file_name):
    file_r = open('%s' % file_name, 'r', encoding='utf-8')
    csv_r = csv.reader(file_r, delimiter='\t', quoting=csv.QUOTE_NONE)
    pool = {}
    standings = {}
    count = 0
    for row in csv_r:
        count += 1
        points = float(row[0])
        gamer = row[1]
        strategy = row[2]
        pool[gamer] = strategy
        standings[gamer] = points
    return(count, pool, standings)


# Функция делает сводную таблицу по проведенным турнирам
def summary(shares, count, count_model, pool_model, output_name):
    # Создаем общий словарь, в который будем добавлять записи из файлов
    # В строках отдельные турниры, в столбцах победители и модельные стратегии
    tournament_dict = {}
    fieldnames = ['tournament', 'gamer', 'strategy', 'points']

    # Обходим в цикле модельные стратегии и добавляем столбцы в таблицу
    gamers_list = sorted(pool_model.items())
    for gamer, strategy in gamers_list:
        fieldnames.append(gamer + ', points')
    for gamer, strategy in gamers_list:
        fieldnames.append(gamer + ', rank')

    # Обходим в цикле турниры
    for share in shares:
        tournament = str(share)  # превращаем описание турнира в строку
        tournament_dict[tournament] = {'tournament': tournament}
        count_r = int(count * share[0] / 100)
        count_n = int(count * share[1] / 100)
        count_m = int(count * share[2] / 100)
        count_total = count_r + count_n + count_m + count_model

        dop_name = '(all=%d, model=%d, r=%d, n=%d, m=%d)' % (
            count_total, count_model, count_r, count_n, count_m)
        file_name = 'Турнир %s.tsv' % dop_name
        try:
            file_r = open('%s' % file_name, 'r', encoding='utf-8')
        except FileNotFoundError:
            continue
        csv_r = csv.reader(file_r, delimiter='\t', quoting=csv.QUOTE_NONE)
        rank = 0
        for row in csv_r:
            rank += 1
            points = float(row[0])
            gamer = row[1]
            strategy = row[2]
            if rank == 1:
                tournament_dict[tournament]['gamer'] = gamer
                tournament_dict[tournament]['strategy'] = strategy
                tournament_dict[tournament]['points'] = points

            if gamer in pool_model:
                tournament_dict[tournament][gamer + ', points'] = points
                tournament_dict[tournament][gamer + ', rank'] = rank
        file_r.close()
        print('done tournament %s' % share)

    file_w = open(output_name, 'w', encoding='utf-8')
    csv_w = csv.DictWriter(file_w, fieldnames, delimiter='\t')
    csv_w.writeheader()
    for tournament in sorted(tournament_dict):
        csv_w.writerow(tournament_dict[tournament])
    file_w.close()

if __name__ == '__main__':
    #  Создаем пул модельных стратегий
    pool_model = {
        'model00': [11, 11, 11, 11, 11, 11, 11, 11, 12],
        'model10': [0, 12, 12, 12, 12, 13, 13, 13, 13],
        'model11': [1, 12, 12, 12, 12, 12, 13, 13, 13],
        'model12': [2, 12, 12, 12, 12, 12, 12, 13, 13],
        'model13': [3, 12, 12, 12, 12, 12, 12, 12, 13],
        'model14': [4, 12, 12, 12, 12, 12, 12, 12, 12],
        'model15': [5, 11, 12, 12, 12, 12, 12, 12, 12],
        'model16': [6, 11, 11, 12, 12, 12, 12, 12, 12],
        'model17': [7, 11, 11, 11, 12, 12, 12, 12, 12],
        'model18': [8, 11, 11, 11, 11, 12, 12, 12, 12],
        'model19': [9, 11, 11, 11, 11, 11, 12, 12, 12],
        'model20': [0, 0, 14, 14, 14, 14, 14, 15, 15],
        'model21': [1, 1, 14, 14, 14, 14, 14, 14, 14],
        'model22': [2, 2, 13, 13, 14, 14, 14, 14, 14],
        'model23': [3, 3, 13, 13, 13, 13, 14, 14, 14],
        'model24': [4, 4, 13, 13, 13, 13, 13, 13, 14],
        'model25': [5, 5, 12, 13, 13, 13, 13, 13, 13],
        'model26': [6, 6, 12, 12, 12, 13, 13, 13, 13],
        'model27': [7, 7, 12, 12, 12, 12, 12, 13, 13],
        'model28': [8, 8, 12, 12, 12, 12, 12, 12, 12],
        'model29': [9, 9, 11, 11, 12, 12, 12, 12, 12],
        'model30': [0, 0, 0, 16, 16, 17, 17, 17, 17],
        'model31': [1, 1, 1, 16, 16, 16, 16, 16, 17],
        'model32': [2, 2, 2, 15, 15, 16, 16, 16, 16],
        'model33': [3, 3, 3, 15, 15, 15, 15, 15, 16],
        'model34': [4, 4, 4, 14, 14, 15, 15, 15, 15],
        'model35': [5, 5, 5, 14, 14, 14, 14, 14, 15],
        'model36': [6, 6, 6, 13, 13, 14, 14, 14, 14],
        'model37': [7, 7, 7, 13, 13, 13, 13, 13, 14],
        'model38': [8, 8, 8, 12, 12, 13, 13, 13, 13],
        'model39': [9, 9, 9, 12, 12, 12, 12, 12, 13],
        'model40': [0, 0, 0, 0, 20, 20, 20, 20, 20],
        'model41': [1, 1, 1, 1, 19, 19, 19, 19, 20],
        'model42': [2, 2, 2, 2, 18, 18, 18, 19, 19],
        'model43': [3, 3, 3, 3, 17, 17, 18, 18, 18],
        'model44': [4, 4, 4, 4, 16, 17, 17, 17, 17],
        'model45': [5, 5, 5, 5, 16, 16, 16, 16, 16],
        'model46': [6, 6, 6, 6, 15, 15, 15, 15, 16],
        'model47': [7, 7, 7, 7, 14, 14, 14, 15, 15],
        'model48': [8, 8, 8, 8, 13, 13, 14, 14, 14],
        'model49': [9, 9, 9, 9, 12, 13, 13, 13, 13],
        'model50': [0, 0, 0, 0, 0, 25, 25, 25, 25],
        'model51': [1, 1, 1, 1, 1, 23, 24, 24, 24],
        'model52': [2, 2, 2, 2, 2, 22, 22, 23, 23],
        'model53': [3, 3, 3, 3, 3, 21, 21, 21, 22],
        'model54': [4, 4, 4, 4, 4, 20, 20, 20, 20],
        'model55': [5, 5, 5, 5, 5, 18, 19, 19, 19],
        'model56': [6, 6, 6, 6, 6, 17, 17, 18, 18],
        'model57': [7, 7, 7, 7, 7, 16, 16, 16, 17],
        'model58': [8, 8, 8, 8, 8, 15, 15, 15, 15],
        'model59': [9, 9, 9, 9, 9, 13, 14, 14, 14],
        'model60': [0, 0, 0, 0, 0, 0, 33, 33, 34],
        'model61': [1, 1, 1, 1, 1, 1, 31, 31, 32],
        'model62': [2, 2, 2, 2, 2, 2, 29, 29, 30],
        'model63': [3, 3, 3, 3, 3, 3, 27, 27, 28],
        'model64': [4, 4, 4, 4, 4, 4, 25, 25, 26],
        'model65': [5, 5, 5, 5, 5, 5, 23, 23, 24],
        'model66': [6, 6, 6, 6, 6, 6, 21, 21, 22],
        'model67': [7, 7, 7, 7, 7, 7, 19, 19, 20],
        'model68': [8, 8, 8, 8, 8, 8, 17, 17, 18],
        'model69': [9, 9, 9, 9, 9, 9, 15, 15, 16],
        'dr01': [12, 11, 11, 11, 11, 11, 11, 11, 11],
        'dr02': [12, 12, 12, 12, 12, 12, 12, 12, 4],
        'dr03': [20, 20, 20, 20, 20, 0, 0, 0, 0],
        'dr04': [21, 21, 21, 21, 12, 1, 1, 1, 1],
        'dr05': [1, 1, 14, 14, 14, 14, 14, 14, 14],
        'dr06': [0, 0, 0, 20, 20, 20, 20, 20, 0],
        'dr07': [0, 20, 20, 0, 0, 20, 20, 0, 20],
        'dr08': [21, 21, 21, 21, 3, 3, 3, 3, 4],
        'dr09': [22, 2, 22, 2, 22, 2, 22, 3, 3],
        'dr10': [21, 2, 15, 2, 21, 14, 21, 2, 2],
        'dr11': [2, 14, 2, 2, 2, 21, 15, 21, 21],
        'dr12': [3, 21, 3, 21, 4, 21, 3, 21, 3],
        'dr13': [21, 21, 21, 12, 13, 3, 3, 3, 3],
        'dr14': [0, 0, 4, 16, 16, 16, 16, 16, 16],
        'dr15': [1, 24, 1, 24, 1, 24, 1, 23, 1],
        'dr16': [20, 10, 10, 10, 10, 10, 10, 10, 10],
        'dr17': [6, 9, 11, 3, 2, 13, 21, 16, 19],
        'dr18': [9, 10, 11, 10, 11, 12, 11, 12, 14],
        'dr19': [11, 11, 11, 11, 15, 11, 10, 10, 10],
        'dr20': [17, 2, 16, 2, 15, 14, 16, 16, 2],
        'dr21': [17, 4, 17, 4, 17, 4, 17, 3, 17],
        'dr22': [3, 2, 1, 15, 45, 16, 15, 3, 0],
        'dr23': [3, 3, 3, 17, 15, 13, 16, 17, 13],
        'dr24': [22, 2, 21, 2, 24, 2, 23, 2, 2],
        'dr25': [8, 8, 9, 10, 11, 12, 13, 14, 15],
        'dr26': [13, 14, 6, 14, 6, 14, 13, 14, 6],
    }
    count_model = len(pool_model)  # количество модельных стратегий

    count = 10000  # общее количество сгенерированных стратегий
    shares = []  # список долей стратегий в каждом турнире
    for r in list(range(0, 101, 10)):
        for n in range(0, 100 - r + 1, 10):
            m = 100 - r - n
            shares.append([r, n, m])

    for share in shares:  # проводим турнир для каждого соотношения долей
        count_r = int(count * share[0] / 100)
        count_n = int(count * share[1] / 100)
        count_m = int(count * share[2] / 100)
        count_total = count_r + count_n + count_m + count_model

        # Определяем распределение макс-минных стратегий
        counts_m = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
        }
        for i in range(count_m):
            m = random.randint(1, 6)
            counts_m[m] += 1

        # Создаем пулы для каждой стратегии
        pool_m = {}
        for m in counts_m:
            pool_m[m] = generate(
                counts_m[m], generator='max-min', n_min_cells=m, max_in_min_cell=6)
        pool_n = generate(count_n, generator='norm', mu=11, sigma=3)
        pool_r = generate(count_r, generator='random')

        # Объединяем все пулы стратегий в общий турнирный пул
        pool = {}
        for m in pool_m:
            for i in pool_m[m]:
                pool['m' + str(m) + str(i)] = pool_m[m][i]
        for n in pool_n:
            pool['n' + str(n)] = pool_n[n]
        for r in pool_r:
            pool['r' + str(r)] = pool_r[r]
        for i in pool_model:
            pool[i] = pool_model[i]

        standings = tournament(pool)  # проводим турнир

        dop_name = '(all=%d, model=%d, r=%d, n=%d, m=%d)' % (
            count_total, count_model, count_r, count_n, count_m)
        file_name = 'Турнир %s.tsv' % dop_name
        save_result(file_name, standings, pool)
        print('done tournament %s' % share)
    print('done all')

    # Создаем сводную таблицу по всем турнирам
    output_name = 'Total (%d).tsv' % (count + count_model)
    summary(shares, count, count_model, pool_model, output_name)
    print('done summary')
