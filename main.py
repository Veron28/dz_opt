import time

t = time.monotonic()

from init_data import FlyData
import numpy as np
from operator import itemgetter
import glob

import warnings
warnings.filterwarnings('ignore')

#init data

xls_files = glob.glob('*.xls')

if len(xls_files) == 0:
    print('Не найдено никаких файлов расширения .xls')
elif len(xls_files) != 1:
    print('Найдено несколько файлов расширения .xls, удалите лишние')
else:
    imp = FlyData(xls_files[0])

    vectors = np.array(imp.df_links[imp.ideal.columns])
    m = imp.ideal.shape[-1] #размерность
    k = imp.ideal.shape[0] #количество групп
    n = imp.df_links.shape[0] #количество векторов
    weights = np.array(imp.weight)[0] # веса для цф
    ideal_vector = np.array(imp.ideal)

    #params

    p = 10 # количество особей в популяции
    r = 10 # параметр для кроссовера
    mu = 0.1 # параметр для мутации
    p_trans = 2*p # количество особей в промежуточной популяции
    cross_part = 0.85 # доля нового поколения, порожденная кроссовером
    nparents = int(p_trans * cross_part)

    # Representation and Initial population

    individual = np.random.randint(0, k, n) # особь

    #По списку всех векторов и вектору распределений по группам, получим списки групп
    def get_groups(vectors, individual):
        num_vect = len(vectors)
        groups = [[vectors[i] for i in range(num_vect) if individual[i] == group] for group in range(k)]
        return np.array(groups)

    # оформим как класс
    class Individ:
        def __init__(self, value, vectors):
            self.value = value
            self.vectors = vectors
        def fitness(self):
            groups = get_groups(self.vectors, self.value)
            #groups = np.array([[self.vectors[i] for i in range(len(self.vectors)) if self.value[i] == group] for group in range(k)])
            of = 0
            i = 0
            for group in groups:
                s = sum(group)
                of += sum(weights * ((1 - s/ideal_vector[i]) ** 2))
                i += 1
            return of

    # groups = get_groups(vectors, individual)
    #
    # ind = Individ(individual, vectors)
    # ind.fitness()

    # Initial solution

    #Жадно добавляет элемент в конец уже выбранным векторам
    def greedy_addition(individual, vectors, el):
        cur_ = Individ(individual, vectors).fitness()
        group_number = individual[el]
        min_fv = cur_
        for group in range(k):
            individual[el] = group
            fv = Individ(individual, vectors).fitness()
            if fv < min_fv:
                min_fv = fv
                group_number = group
        if min_fv < cur_:
            individual[el] = group_number
        return individual

    #Жадно заполняет вектор до конца начиная с элемента start
    def greedy_fill(individual, vectors, start):
        for el in range(start, n):
            individual = greedy_addition(individual, vectors, el)
        return individual

    # строим начальную популяцию
    def get_initial_population(start):
        population_values = np.random.randint(0, k, (p,n))
        population = list(map(lambda x: Individ(greedy_fill(x, vectors, start), vectors), population_values))
        return population

    # population = get_initial_population(n)
    #Genetic operators¶
    def get_candidates(population):
        candidates_index = np.random.choice(range(len(population)), r)
        candidates = itemgetter(*candidates_index)(population)
        return candidates


    def choose_parent(population, candidates):
        # выбираем кандидатов
        # ----candidates_index = np.random.choice(range(len(population)), r)
        # ----candidates = itemgetter(*candidates_index)(population)


        # бросаем монетку
        nu = np.random.randint(0, 2)
        if nu == 1:  # если единица, то случайный
            p = candidates[np.random.randint(0, r)]
        else:  # иначе лучший
            p = max(candidates, key=lambda i: i.fitness())
        return p

    def crossover(p1, p2):
        y = np.random.randint(0, m)
        ch1 = Individ(np.concatenate((p1.value[:y], p2.value[y:]), axis=0), vectors)
        ch2 = Individ(np.concatenate((p2.value[:y], p1.value[y:]), axis=0), vectors)
        return ch1, ch2

    # p1, p2 = choose_parent(), choose_parent()
    # ch1, ch2 = crossover(p1, p2)

    def mutation(ind):
        val = ind.value
        for i in range(len(val)):
            nu = np.random.random()
            if nu <= mu:
                new_c = np.random.randint(0, k)
                val[i] = new_c
        return Individ(val, vectors)

    # candidates = get_candidates()
    # ind = candidates[np.random.randint(0,r)]
    # new_ind = mutation(ind)

    # New generation

    # дана популяция
    # population_values = np.random.randint(0, k, (p,n))
    # population = [Individ(val, vectors) for val in population_values]


    def generate_population(population):
        new_population = []
        candidates = get_candidates(population)
        for _ in range(nparents // 2):
            p1, p2 = choose_parent(population, candidates), choose_parent(population, candidates)
            ch1, ch2 = crossover(p1, p2)
            new_population.append(ch1)
            new_population.append(ch2)

        for _ in range(int(p_trans - len(new_population))):
            ind = candidates[np.random.randint(0, r)]
            new_ind = mutation(ind)
            new_population.append(new_ind)

        return new_population

    # new_population = generate_population(population)


    # Selection

    # population = sorted(new_population, key = lambda i: i.fitness() )[:p]

    # Local improvement procedure

    def one_change(ind):
        # выбираем случайную позицию
        val = ind.value
        pos = np.random.randint(0, n)
        old_group = val[pos]

        # выбираем новую группу
        a = list(range(k))
        a.remove(old_group)
        new_group = np.random.choice(a)

        # формируем нового индивида
        new_val = val.copy()
        new_val[pos] = new_group
        new_ind = Individ(new_val, vectors)

        # сравниваем цф
        if new_ind.fitness() < ind.fitness():
            return new_ind
        else:
            return ind

    # old = population[1]
    # new = one_change(old)


    def two_change(ind):
        # выбираем две случайные позиции
        val = ind.value
        pos1, pos2 = np.random.randint(0, n, 2)
        g1, g2 = val[pos1], val[pos2]

        # формируем нового индивида
        new_val = val.copy()
        new_val[pos1] = g2
        new_val[pos2] = g1
        new_ind = Individ(new_val, vectors)

        # сравниваем цф
        if new_ind.fitness() < ind.fitness():
            return new_ind
        else:
            return ind

    # old = population[2]
    # new = one_change(old)


    def three_change(ind):
        # выбираем три случайные позиции
        val = ind.value
        pos1, pos2, pos3 = np.random.randint(0, n, 3)
        g1, g2, g3 = val[pos1], val[pos2], val[pos3]

        # формируем нового индивида
        new_val = val.copy()
        new_val[pos1] = g2
        new_val[pos2] = g3
        new_val[pos3] = g1
        new_ind = Individ(new_val, vectors)

        # сравниваем цф
        if new_ind.fitness() < ind.fitness():
            return new_ind
        else:
            return ind

    # old = population[3]
    # new = one_change(old)

    # все вместе
    def local_search(ind):
        ind = one_change(ind)
        ind = two_change(ind)
        ind = three_change(ind)
        return ind

    # old = population[3]
    # new = local_search(old)

    iter_n = 10  # количество итераций алгоритма
    # начальная популяция генерируется с помощью случайной генерации + жадного алгоритма

    population = get_initial_population(n)



    best = max(population, key=lambda i: i.fitness())
    fitness_vals = [best.fitness()]

    while time.monotonic() - t < 60 * 3 - 3:
        # генерируем новую популяцию
        new_population = generate_population(population)
        population = sorted(new_population, key=lambda i: i.fitness())[:p]

        # улучшаем новую популяцию
        population = [local_search(ind) for ind in population]

        # давайте на каждом шаге смотреть, какой будет фитнес у лучшего представителя
        best = max(population, key=lambda i: i.fitness())
        best_f = best.fitness()

        # критерий останова
        # if abs(best_f - fitness_vals[-1]) < 10 ** (-5):
        #     break
        # else:
        fitness_vals.append(best_f)


    imp.df['Группа'] = best.value + 1
    imp.df.to_excel('ans.xlsx')
    print('Результат записан в файл ans.xlsx')





