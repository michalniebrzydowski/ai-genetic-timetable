import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools, algorithms


class Teacher:
    def __init__(self, id, name, courses):
        self.id = id
        self.name = name
        self.courses = courses  # list of Course ids


class Course:
    def __init__(self, id, name, times_per_week):
        self.id = id
        self.name = name
        self.times_per_week = times_per_week


class Classroom:
    def __init__(self, id, name):
        self.id = id
        self.name = name


TEACHERS = [Teacher(0, 'Anna Matematyczna', [0]),
            Teacher(1, 'Aniela Anielska', [1]),
            Teacher(2, 'Krzysztof Fizyczny', [2]),
            Teacher(3, 'Adam Geograficzny', [3]),
            Teacher(4, 'Michal Histeryk', [4]),
            Teacher(5, 'Daniel Chemiczny', [5])]

COURSES = [Course(0, 'Matematyka', 12),
           Course(1, 'Angielski', 12),
           Course(2, 'Fizyka', 12),
           Course(3, 'Geografia', 12),
           Course(4, 'Historia', 12),
           Course(5, 'Chemia', 12)]

CLASSROOMS = [Classroom(0, 'Sala 1'),
              Classroom(1, 'Sala 2'),
              Classroom(2, 'Sala 3'),
              Classroom(3, 'Sala 4')]

DAYS_OF_WEEK = 5
TIME_SLOTS = 8


def evaluate(individual):
    # Convert to numpy for convenience
    schedule = np.array(individual)

    conflicts = 0

    # Check if all courses are assigned the correct number of times
    assigned_courses = schedule[:, 3]  # Course ids
    assigned_counts = np.bincount(assigned_courses)  # Count occurrences of each course
    for i, course in enumerate(COURSES):
        if assigned_counts[i] != course.times_per_week:
            conflicts += (10 * (course.times_per_week - assigned_counts[i]))

    # Check if a teacher is assigned to two different courses at the same time
    for i in range(DAYS_OF_WEEK):
        for j in range(TIME_SLOTS):
            for t in range(len(TEACHERS)):
                # Get the indices of the courses taught by this teacher at this time
                indices = np.where((schedule[:, 0] == t) & (schedule[:, 1] == i) & (schedule[:, 2] == j))[0]
                if len(indices) > 1:
                    conflicts += 10

    # Check if a classroom has more than one course at the same time
    for i in range(DAYS_OF_WEEK):
        for j in range(TIME_SLOTS):
            for c in range(len(CLASSROOMS)):
                # Get the indices of the courses scheduled in this classroom at this time
                indices = np.where((schedule[:, 4] == c) & (schedule[:, 1] == i) & (schedule[:, 2] == j))[0]
                if len(indices) > 1:
                    conflicts += 10

    return conflicts,


def create_individual():
    individual = []
    assigned_courses = []
    for course in COURSES:
        # Get the teachers who can teach this course
        available_teachers = [teacher.id for teacher in TEACHERS if course.id in teacher.courses]
        for _ in range(course.times_per_week):
            # Choose a random teacher, day, time, and classroom
            teacher_id = random.choice(available_teachers)
            day = random.randint(0, DAYS_OF_WEEK - 1)
            time = random.randint(0, TIME_SLOTS - 1)
            classroom_id = random.randint(0, len(CLASSROOMS) - 1)
            individual.append([teacher_id, day, time, course.id, classroom_id])
            assigned_courses.append(course.id)

    return individual


def crossover(ind1, ind2):
    return tools.cxTwoPoint(ind1, ind2)


def mutation(individual):
    # Choose a random gene
    gene_idx = random.randint(0, len(individual) - 1)

    # Mutation 1: Changing teacher
    course_id = individual[gene_idx][3]
    available_teachers = [teacher.id for teacher in TEACHERS if course_id in teacher.courses]
    individual[gene_idx][0] = random.choice(available_teachers)

    # Mutation 2: Changing day
    individual[gene_idx][1] = random.randint(0, DAYS_OF_WEEK - 1)

    # Mutation 3: Changing time
    individual[gene_idx][2] = random.randint(0, TIME_SLOTS - 1)

    # Mutation 4: Changing Classroom
    individual[gene_idx][4] = random.randint(0, len(CLASSROOMS) - 1)

    return individual,


# Set up the genetic algorithm
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, create_individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', evaluate)
toolbox.register('mate', crossover)
toolbox.register('mutate', mutation)
toolbox.register('select', tools.selTournament, tournsize=3)


def run_genetic_algorithm():
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    algorithms_list = [
        ('eaMuCommaLambda_50_100', partial(algorithms.eaMuCommaLambda, mu=50, lambda_=100)),
        ('eaSimple', partial(algorithms.eaSimple))
    ]

    results = []  # Best results

    for algorithm_name, algo in algorithms_list:
        pop = toolbox.population(n=100)
        pop, logbook = algo(pop, toolbox, cxpb=0.3, mutpb=0.2, ngen=100, stats=stats, halloffame=hof)
        best_schedule = hof[0]
        create_schedule_image(best_schedule, algorithm_name)
        num_classes = len(best_schedule)
        results.append((algorithm_name, num_classes))

        # Generate statistics
        best_individual = tools.selBest(pop, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        print(f"Best individual with fitness: {best_fitness}")
        print(f"Algorithm: {algorithm_name}")
        print(f"Number of conflicts: {evaluate(best_individual)[0]}")
        print(f"Total number of classes: {num_classes}")
        print("\n")

        # Generate effectiveness summary
        gen = logbook.select("gen")
        fit_avg = logbook.select("avg")
        fit_min = logbook.select("min")
        fit_max = logbook.select("max")

        plt.figure()
        plt.plot(gen, fit_avg, label="avg")
        plt.plot(gen, fit_min, label="min")
        plt.plot(gen, fit_max, label="max")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(f"Fitness Progression - {algorithm_name}")
        plt.legend(loc="best")
        plt.savefig(f"fitness_{algorithm_name}.png")

    print("Summary:")
    for result in results:
        algorithm_name, num_classes = result
        print(f"Algorithm: {algorithm_name}, Number of classes: {num_classes}")

    return "Generation and visualization completed."


def create_schedule_image(schedule, algorithm_name):
    # Create an empty schedule for each classroom
    classroom_schedules = [[['' for _ in range(DAYS_OF_WEEK)] for _ in range(TIME_SLOTS)] for _ in
                           range(len(CLASSROOMS))]

    # Fill in the schedules
    for gene in schedule:
        teacher_id, day, time, course_id, classroom_id = gene
        teacher_name = [t.name for t in TEACHERS if t.id == teacher_id][0]
        course_name = [c.name for c in COURSES if c.id == course_id][0]
        classroom_schedules[classroom_id][time][day] = f'{course_name}\n{teacher_name}\n'

    # Define time slots and day labels
    time_slots_labels = [f'Time slot {i + 1}' for i in range(TIME_SLOTS)]
    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Create the plots for each classroom
    for classroom_id, classroom_schedule in enumerate(classroom_schedules):
        fig, axs = plt.subplots(figsize=(10, 10))
        axs.axis('tight')
        axs.axis('off')
        table = axs.table(cellText=classroom_schedule, cellLoc='center', loc='center', colLabels=day_labels,
                          rowLabels=time_slots_labels)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.4, 2.2)
        plt.tight_layout()
        plt.savefig(f'schedule_{algorithm_name}_classroom_{classroom_id}.png')


run_genetic_algorithm()
