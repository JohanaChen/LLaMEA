"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""
import concurrent.futures
import contextlib
import logging
import os
import random
import re
import traceback
import json

import numpy as np
from ConfigSpace import ConfigurationSpace
from joblib import Parallel, delayed

from .loggers import ExperimentLogger
from .solution import Solution
from .utils import NoCodeException, discrete_power_law_distribution, handle_timeout

from HIIT_maker.utils.state import _CHOICE_STATE
from HIIT_maker.evaluation.Hybrid import Hybrid


# TODOs:
# Implement diversity selection mechanisms (none, prefer short code, update population only when (distribution of) results is different, AST / code difference)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LLaMEA:
    """
    A class that represents the Language Model powered Evolutionary Algorithm (LLaMEA).
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """

    def __init__(
        self,
        f,
        llm,
        n_parents=5,
        n_offspring=5,
        role_prompt="",
        task_prompt="",
        example_prompt=None,
        output_format_prompt=None,
        experiment_name="",
        elitism=True,
        HPO=False,
        mutation_prompts=None,
        adaptive_mutation=False,
        budget=100,
        eval_timeout=3600,
        max_workers=10,
        parallel_backend="loky",
        log=True,
        minimization=False,
        _random=False,
    ):
        """
        Initializes the LLaMEA instance with provided parameters. Note that by default LLaMEA maximizes the objective.

        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            llm (object): An instance of a language model that will be used to generate and evolve algorithms.
            n_parents (int): The number of parents in the population.
            n_offspring (int): The number of offspring each iteration.
            elitism (bool): Flag to decide if elitism (plus strategy) should be used in the evolutionary process or comma strategy.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            example_prompt (str): An example prompt to guide the language model in generating code (or None for default).
            output_format_prompt (str): A prompt that specifies the output format of the language model's response.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            HPO (bool): Flag to decide if hyper-parameter optimization is part of the evaluation function.
                In case it is, a configuration space should be asked from the LLM as additional output in json format.
            mutation_prompts (list): A list of prompts to specify mutation operators to the LLM model. Each mutation, a random choice from this list is made.
            adaptive_mutation (bool): If set to True, the mutation prompt 'Change X% of the lines of code' will be used in an adaptive control setting.
                This overwrites mutation_prompts.
            budget (int): The number of generations to run the evolutionary algorithm.
            eval_timeout (int): The number of seconds one evaluation can maximum take (to counter infinite loops etc.). Defaults to 1 hour.
            max_workers (int): The maximum number of parallel workers to use for evaluating individuals.
            parallel_backend (str): The backend to use for parallel processing (e.g., 'loky', 'threading').
            log (bool): Flag to switch of the logging of experiments.
            minimization (bool): Whether we minimize or maximize the objective function. Defaults to False.
            _random (bool): Flag to switch to random search (purely for debugging).
        """
        self.llm = llm
        self.model = llm.model
        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.
        self.role_prompt = role_prompt
        self.parallel_backend = parallel_backend
        self.example_prompt = example_prompt or ""
        if role_prompt == "":
            self.role_prompt = "You are a professional fitness coach who designs high-quality, balanced HIIT workout programs."
        if task_prompt == "":
            self.task_prompt = """
            Design a high-intensity interval training (HIIT) program in JSON format.
            The program should be 10-40 minutes long in total.Each exercise of the program must include:- The name of the exercise- The duration of the exercise (in seconds)- The rest period after the exercise (in seconds)Include warm-up and cool-down phases and avoid repeating the same exercise too often.
            The core training should include a balance of cardio, strength, and flexibility exercises.
            Ensure variety in terms of range of:
            Types of exercises
            Training structures (e.g. EMOM, AMRAP, E2MOM, Death-by, Tabata, etc.)
            Number of repetitions
            Load and intensity (implicitly controlled via duration and rest).
            Each generated program doesn't have to contain the same set of exercises, different structures of programs are allowed, such as linear sequence, repeated circuits, or block cycles.
            Make the program suitable for intermediate-level individuals with moderate fitness.
            Annotate exercises with a tag like `(type: cardio)` or `(intensity: medium)` to indicate the nature and difficulty of the exercise.
            The output should be pure JSON and easy-to-read list of exercises, for example:
            {
            "warm_up": [
                { "name": "Jumping Jacks", "duration": 30, "rest": 10, "type": "cardio" },
                { "name": "Arm Circles", "duration": 20, "rest": 10, "type": "mobility" }
            ],
            "main_set": {
                "type": "EMOM",
                "rounds": 5,
                "exercises": [
                { "name": "Burpees", "duration": 40, "rest": 20, "type": "cardio", "intensity": "high" },
                { "name": "Push-ups", "duration": 30, "rest": 30, "type": "strength", "intensity": "medium" },
                { "name": "Jump Squats", "duration": 30, "rest": 20, "type": "strength/cardio" }
                ]
            },
            "cool_down": [
                { "name": "Forward Fold", "duration": 30, "rest": 10, "type": "flexibility" },
                { "name": "Child's Pose", "duration": 30, "rest": 10, "type": "recovery" }
            ]
            }
        """
        else:
            self.task_prompt = task_prompt

        self.output_format_prompt = """
        Only return the HIIT training program in JSON. 
        Do not include Python code, variable names, markdown, or explanations.

        Use the following formatting:

        {
            "warm_up": [
                { "name": "Jumping Jacks", "duration": 30, "rest": 10, "type": "cardio" },
                { "name": "Arm Circles", "duration": 20, "rest": 10, "type": "mobility" }
            ],
            "main_set": {
                "type": "EMOM",
                "rounds": 5,
                "exercises": [
                { "name": "Burpees", "duration": 40, "rest": 20, "type": "cardio", "intensity": "high" },
                { "name": "Push-ups", "duration": 30, "rest": 30, "type": "strength", "intensity": "medium" },
                { "name": "Jump Squats", "duration": 30, "rest": 20, "type": "strength/cardio" }
                ]
            },
            "cool_down": [
                { "name": "Forward Fold", "duration": 30, "rest": 10, "type": "flexibility" },
                { "name": "Child's Pose", "duration": 30, "rest": 10, "type": "recovery" }
            ]
            }
        """
       
        self.mutation_prompts = mutation_prompts
        self.adaptive_mutation = adaptive_mutation
        if mutation_prompts == None:
            self.mutation_prompts = [
                # "Refine the strategy of the selected solution to improve it.",  # small mutation
                # "Generate a new HIIT program that is different from the programs you have tried before.", #new random solution
                "Generate a new HIIT program that explores a different style, intensity distribution, or exercise selection compared to previous programs."
                # "Generate a complete different HIIT program with completely different style, intensity distribution, and exercise selection compared to previous programs."
            ]
        self.budget = budget
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.population = []
        self.elitism = elitism
        self.generation = 0
        self.run_history = []
        self.log = log
        self._random = _random
        self.HPO = HPO
        self.minimization = minimization
        self.worst_value = -np.Inf
        if minimization:
            self.worst_value = np.Inf
        self.best_so_far = Solution(name="", data={})
        self.best_so_far.set_scores(self.worst_value, "", "")
        self.experiment_name = experiment_name

        if self.log:
            modelname = self.model.replace(":", "_")
            self.logger = ExperimentLogger(f"LLaMEA-{modelname}-{experiment_name}")
            self.llm.set_logger(self.logger)
        else:
            self.logger = None
        self.textlog = logging.getLogger(__name__)
        if max_workers > self.n_offspring:
            max_workers = self.n_offspring
        self.max_workers = max_workers
        self.choice_state=_CHOICE_STATE

    def logevent(self, event):
        self.textlog.info(event)

    def initialize_single(self):
        """
        Initializes a single solution (HIIT program).
        """
        new_individual = Solution(name="", data={}, generation=self.generation)
        session_messages = [
            {
                "role": "user",
                "content": self.role_prompt
                + self.task_prompt
                + self.example_prompt
                + self.output_format_prompt,
            },
        ]
        try:
            new_individual = self.llm.sample_solution(session_messages, HPO=self.HPO)
            new_individual.generation = self.generation

            # Save the text to a .txt file
            # hiit_text = new_individual.text
            # filename = f"hiit_gen_{self.generation}_id_{random.randint(1000, 9999)}.txt"
            # with open(filename, "w", encoding="utf-8") as f:
            #     f.write(hiit_text)
            # # Evaluate the program's fitness based on text
            # new_individual = self.evaluate_fitness(new_individual)

            # Convert LLM JSON string to dict
            hiit_data = new_individual.data  # <--- convert JSON string to Python dict
            hiit_data.update({
                "generation": self.generation,
                "id": random.randint(1000, 9999),
                "name": new_individual.name,
            })

            # Save to JSON file
            filename = f"hiit_gen_{hiit_data['generation']}_id_{hiit_data['id']}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(hiit_data, f, indent=4, ensure_ascii=False)

            # Evaluate fitness
            new_individual = self.evaluate_fitness(new_individual)

        except Exception as e:
            new_individual.set_scores(
                self.worst_value,
                f"An exception occured: {traceback.format_exc()}.",
                repr(e) + traceback.format_exc(),
            )
            self.logevent(f"An exception occured: {traceback.format_exc()}.")
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(new_individual)

        return new_individual

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """

        population = []
        population_gen = []
        try:
            timeout = self.eval_timeout
            population_gen = Parallel(
                n_jobs=self.max_workers,
                backend=self.parallel_backend,
                timeout=timeout + 15,
                return_as="generator_unordered",
            )(delayed(self.initialize_single)() for _ in range(self.n_parents))
        except Exception as e:
            print(f"Parallel time out in initialization {e}, retrying.")

        for p in population_gen:
            self.run_history.append(p)  # update the history
            population.append(p)

        self.generation += 1
        self.population = population  # Save the entire population
        if not population:
            raise RuntimeError("Failed to initialize population. No individuals were generated.")
        self.update_best()

    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of the provided individual by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            individual (Solution): The solution instance to evaluate.

        Returns:
            Solution: The updated solution with feedback, fitness and error information filled in.
        """
        # with contextlib.redirect_stdout(None):
        #     updated_individual = self.f(individual, self.logger)
        updated_individual = self.f(individual, self.logger)
        return updated_individual

    def construct_prompt(self, individual):
        """
        Constructs a new session prompt for the language model based on a selected HIIT plan.
        """
        # Generate the current population summary
        population_summary = "\n".join([ind.get_summary() for ind in self.population])
        # solution_text = individual.text
        # Parse the previous HIIT solution (convert from JSON string to Python dict)
        try:
            if isinstance(individual.data, str):
                hiit_data = json.loads(individual.data)  # si es string, intenta parsearlo como JSON
            else:
                hiit_data = individual.data              # si ya es dict/list, úsalo tal cual
        except Exception:
            # si no se puede parsear, mételo como texto crudo
            hiit_data = {"raw_text": str(individual.data)}

        description = individual.description
        feedback = individual.feedback
        last_feedback = individual.last_feedback

        if self.adaptive_mutation:
            # num_lines = len(solution_text.splitlines())
            num_lines = len(individual.splitlines())
            prob = discrete_power_law_distribution(num_lines, 1.5) # Follows the discrete power-law distribution to pick a probability of change.
            lines_to_change = max (1, int(prob*num_lines)) # Ensures changing at least 1 line of the program.
            new_mutation_prompt = (f"Improve this HIIT plan by changing about {lines_to_change} lines ({(prob*100):.1f}% of the content). Keep the rest unchanged")
            self.mutation_prompts = [new_mutation_prompt]

        mutation_operator = random.choice(self.mutation_prompts)
        individual.set_operator(mutation_operator)

        # Prepare the JSON plan as a formatted string for the prompt
        hiit_json_string = json.dumps(hiit_data, indent=4, ensure_ascii=False)

        final_prompt = f"""{self.task_prompt}
The current HIIT program to improve is (in JSON format):
{hiit_json_string}

With feedback:
{feedback}

With user feedback:
{last_feedback}

{mutation_operator}

{self.output_format_prompt}
"""
        # if getattr(individual, "last_feedback", None):
        #     final_prompt += f"\n\nUser feedback from the last round: {individual.last_feedback}"
        # elif hasattr(self, "choice_state") and self.choice_state.get("last_feedback"):
        #     final_prompt += f"\n\nUser feedback from the last round: {self.choice_state['last_feedback']}"

        # 🔍 System feedback (ya lo tienes en individual.feedback)
        system_fb = getattr(individual, "feedback", None)

        # 🔍 User feedback: primero intento en el objeto, luego fallback al estado global
        user_fb = getattr(individual, "last_feedback", None)
        if not user_fb and self.choice_state:
            user_fb = self.choice_state.get("incumbent_feedback")

        # Añadir ambos feedbacks al prompt
        final_prompt += f"\n\nSystem feedback: {system_fb}"
        if user_fb:
            final_prompt += f"\n\nUser feedback from the last round: {user_fb}"

        session_messages = [
            {"role": "user", "content": self.role_prompt + final_prompt},
        ]

        if self._random:  # not advised to use, only for debugging purposes
            session_messages = [
                {"role": "user", "content": self.role_prompt + self.task_prompt},
            ]

        # Logic to construct the new prompt based on current evolutionary state.
        return session_messages

    def update_best(self):
        """
        Update the best individual in the new population
        """
        if self.minimization == False:
            best_individual = max(self.population, key=lambda x: x.fitness)

            if best_individual.fitness > self.best_so_far.fitness:
                self.best_so_far = best_individual
        else:
            best_individual = min(self.population, key=lambda x: x.fitness)

            if best_individual.fitness < self.best_so_far.fitness:
                self.best_so_far = best_individual

    def selection(self, parents, offspring):
        """
        Select the new population based on the parents and the offspring and the current strategy.

        Args:
            parents (list): List of solutions.
            offspring (list): List of new solutions.

        Returns:
            list: List of new selected population.
        """
        reverse = self.minimization == False

        # TODO filter out non-diverse solutions
        if self.elitism:
            # Combine parents and offspring
            combined_population = parents + offspring
            # Sort by fitness
            combined_population.sort(key=lambda x: x.fitness, reverse=reverse)
            # Select the top individuals to form the new population
            new_population = combined_population[: self.n_parents]
        else:
            # Sort offspring by fitness
            offspring.sort(key=lambda x: x.fitness, reverse=reverse)
            # Select the top individuals from offspring to form the new population
            new_population = offspring[: self.n_parents]

        return new_population

    def evolve_solution(self, individual):
        """
        Evolves a single HIIT solution by constructing a new prompt,
        querying the LLM, and evaluating the fitness.
        """
        # new_prompt = self.construct_prompt(individual)
        # # 🔥 Add user feedback if available
        # if getattr(individual, "last_feedback", None):
        #     print("The feedback from the user is: " + individual.last_feedback) #DEBUG
        #     new_prompt += f"\n\nUser feedback from the last round: {individual.last_feedback}"
        # else:
        #     print("No feedback passed for this individual.")

        session_messages = self.construct_prompt(individual)

        # 🔒 Normalización defensiva: aceptar str, dict o lista mixta
        if isinstance(session_messages, str):
            session_messages = [{"role": "user", "content": session_messages}]
        elif isinstance(session_messages, dict):
            session_messages = [session_messages]
        elif isinstance(session_messages, list):
            norm = []
            for m in session_messages:
                if isinstance(m, dict) and "content" in m:
                    norm.append(m)
                else:
                    # si por error te llega un string u otro tipo dentro de la lista
                    norm.append({"role": "user", "content": str(m)})
            session_messages = norm
        else:
        # Último resort
            session_messages = [{"role": "user", "content": str(session_messages)}]

        evolved_individual = individual.copy()

        # try:
        #     # Evolve a new HIIT program with LLM
        #     evolved_individual = self.llm.sample_solution(
        #         new_prompt, evolved_individual.parent_ids, HPO=self.HPO
        #     )
            
        #     evolved_individual.generation = self.generation
        #     # Evaluate the new HIIT program
        #     evolved_individual = self.evaluate_fitness(evolved_individual)

        try:
            evolved_individual = individual.copy()
            evolved_individual = self.llm.sample_solution(
                session_messages,  # <- PASA LOS MENSAJES NORMALIZADOS
                evolved_individual.parent_ids,
                HPO=self.HPO
            )
            evolved_individual.generation = self.generation
            evolved_individual = self.evaluate_fitness(evolved_individual)
        except Exception as e:
            error = repr(e)
            evolved_individual.set_scores(
                self.worst_value, f"An exception occurred: {error}.", error
            )
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(evolved_individual)
            self.logevent(f"An exception occured: {traceback.format_exc()}.")

        # self.progress_bar.update(1)
        return evolved_individual

    def run(self):
        """
        Main loop to evolve the solutions until the evolutionary budget is exhausted.
        The method iteratively refines solutions through interaction with the language model,
        evaluates their fitness, and updates the best solution found.

        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        # self.progress_bar = tqdm(total=self.budget)
        self.logevent("Initializing first population")
        self.initialize()  # Initialize a population
        # self.progress_bar.update(self.n_parents)

        if self.log:
            self.logger.log_population(self.population)

        self.logevent(
            f"Started evolutionary loop, best so far: {self.best_so_far.fitness}"
        )
        while len(self.run_history) < self.budget:
            # pick a new offspring population using random sampling
            new_offspring_population = np.random.choice(
                self.population, self.n_offspring, replace=True
            )

            new_population = []
            try:
                # Create offspring in parallel 
                timeout = self.eval_timeout
                new_population_gen = Parallel(
                    n_jobs=self.max_workers,
                    timeout=timeout + 15,
                    backend="loky",
                    return_as="generator_unordered",
                )(
                    delayed(self.evolve_solution)(individual)
                    for individual in new_offspring_population
                )
            except Exception as e:
                print("Parallel time out .")

            for p in new_population_gen:
                self.run_history.append(p)
                new_population.append(p)
            self.generation += 1

            if self.log:
                self.logger.log_population(new_population)

            # Update population and the best solution
            self.population = self.selection(self.population, new_population)
            self.update_best()
            self.logevent(
                f"Generation {self.generation}, best so far: {self.best_so_far.fitness}"
            )

        # --- Package useful info before returning ---
        best = self.best_so_far
        best_data = {
            "id": best.id,
            "description": best.description,
            "fitness": best.fitness,
            "generation": best.generation,
            "metadata": best.metadata,
        }
        # If this individual already contains its workout sections (warm_up, main_set, cool_down)
        # inside .data, merge them in:
        if isinstance(best.data, dict):
            # Copy only workout-related sections if they exist
            for section in ["warm_up", "main_set", "cool_down"]:
                if section in best.data:
                    best_data[section] = best.data[section]

        best.data = best_data

        return best
