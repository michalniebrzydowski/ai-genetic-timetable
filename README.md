# School Schedule Optimization

This project applies genetic algorithms to optimize the scheduling of classes in a school. The problem is modeled with teachers, courses, and classrooms. Each teacher can teach specific courses and each course must be taught a certain number of times per week. The goal is to assign each course to a time slot in a specific classroom such that no teacher or classroom is double-booked and all courses are taught the desired number of times.

## Features

1. Definition of teachers, courses, classrooms, and the number of time slots per week.
2. Creation of a population of possible schedules.
3. Evaluation of schedules based on the number of scheduling conflicts.
4. Crossover and mutation operations to create new generations of schedules.
5. Application of the genetic algorithm to find a schedule with the fewest conflicts.
6. Visualization of the best found schedule and the progress of the algorithm.

## Getting Started

The main script is runnable and does not require any additional packages apart from the Python Standard Library and some commonly used packages like numpy, matplotlib and DEAP. If you haven't installed these packages, you can do so using pip:

```
pip install numpy matplotlib deap
```

Then, simply run the script with python:

```
python school_schedule.py
```

## Output

The script will print out the best schedule found, the number of conflicts in this schedule, and some statistics about the performance of the genetic algorithm. It will also create a set of images visualizing the best schedule for each classroom and the progression of the algorithm's fitness over generations.

## Customization

The script can be customized by modifying the definitions of the teachers, courses, classrooms, and time slots, as well as the parameters of the genetic algorithm.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the terms of the MIT (and BUT maybe) license.
