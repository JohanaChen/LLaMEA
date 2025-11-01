import os
import json
import numpy as np
from ioh import get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA, Gemini_LLM, OpenAI_LLM, Ollama_LLM
import logging


# Setup global logger
log = logging.getLogger("evaluate_HIIT")
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    # Execution code starts here

    # GEMINI
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-1.5-flash"
    experiment_name = "pop1-5"
    llm = Gemini_LLM(api_key, ai_model)

    # OPENAI
    # api_key = os.getenv("OPENAI_API_KEY")
    # ai_model = "gpt-3.5-turbo" 
    # experiment_name = "hiit1"
    # llm = OpenAI_LLM(api_key, ai_model)

    # Ollama
    # llm = Ollama_LLM(model="llama3")

    def evaluate_HIIT(solution, logger=None):
        raw_text = solution.text  # <-- use plain text instead of solution.code
        feedback = ""
        try:
            # Extract all lines that look like exercise entries
            lines = raw_text.splitlines()
            program = []

            exercise_pattern = re.compile(
                r"- (.+?) — (?:.*?(\d+)\s*sec.*?|(\d+)\s*reps?.*?), Rest: (\d+)\s*sec",
                re.IGNORECASE
            )

            for line in lines:
                match = exercise_pattern.search(line)
                if match:
                    name = match.group(1).strip()
                    duration = match.group(2)
                    reps = match.group(3)
                    rest = int(match.group(4))

                    if duration:
                        duration = int(duration)
                    elif reps:
                        duration = int(reps)*5 # Suppose each repetition takes 5 seconds.
                    else:
                        duration = 30

                    program.append({
                        "name": name,
                        "duration": duration,
                        "rest": rest
                    })
                else:
                    print(f"⚠️ Skipped line (no match): {line}")

            if not program:
                raise ValueError("No valid exercises found in text.")

            # total_duration = sum(ex["duration"] + ex["rest"] for ex in program)
            total_work = sum(ex["duration"] for ex in program)
            total_rest = sum(ex["rest"] for ex in program)
            total_duration = total_rest + total_work
            num_exercises = len(program)
            unique_exercises = len(set(ex["name"] for ex in program))
            density = min(10, total_work/total_rest*5)

            fitness = 0
            # fitness += -abs(total_duration - 1500)  # Target 25 min (1500 sec)
            fitness += density
            fitness += unique_exercises * 5
            fitness -= sum(1 for ex in program if ex["rest"] > 30) * 10

            # Add these prints to debug
            print("---- Evaluation Debug Info ----")
            print(f"Total duration: {total_duration}s")
            print(f"Number of exercises: {num_exercises}")
            print(f"Unique exercises: {unique_exercises}")
            print(f"Computed fitness: {fitness}")
            print("--------------------------------")

            feedback = f"Parsed {num_exercises} exercises, total duration {total_duration}s, with {unique_exercises} unique exercises."
            solution.set_scores(fitness, feedback)

        except Exception as e:
            feedback = f"Failed to evaluate program: {e}"
            log.error("Evaluation failed: " + feedback)
            solution.set_scores(-9999, feedback)

        return solution
    
    task_prompt = """
    Design a high-intensity interval training (HIIT) program in pure text format.
    The program should be 10-40 minutes long in total.

    Each exercise of the program must include:
    - The name of the exercise
    - The duration of the exercise (in seconds)
    - The rest period after the exercise (in seconds)
    Include warm-up and cool-down phases and avoid repeating the same exercise too often.
    The core training should include a balance of cardio, strength, and flexibility exercises.

    Ensure variety in terms of range of:
    - Types of exercises
    - Training structures (e.g. EMOM, AMRAP, E2MOM, Death-by, Tabata, etc.)
    - Number of repetitions
    - Load and intensity (implicitly controlled via duration and rest).
    Each generated program doesn't have to contain the same set of exercises, different structures of programs are allowed, such as linear sequence, repeated circuits, or block cycles.
    Make the program suitable for intermediate-level individuals with moderate fitness.
    Annotate exercises with a tag like `(type: cardio)` or `(intensity: medium)` to indicate the nature and difficulty of the exercise.

    The output should be pure text and easy-to-read list of exercises, for example:
    Warm-up:
    - Jumping Jacks — 30 sec, Rest: 10 sec (type: cardio)
    - Arm Circles — 20 sec, Rest: 10 sec (type: mobility)

    Main Set (EMOM for 5 rounds):
    - Burpees — 40 sec, Rest: 20 sec (type: cardio, intensity: high)
    - Push-ups — 30 sec, Rest: 30 sec (type: strength, intensity: medium)
    - Jump Squats — 30 sec, Rest: 20 sec (type: strength/cardio)

    Cool-down:
    - Forward Fold — 30 sec, Rest: 10 sec (type: flexibility)
    - Child's Pose — 30 sec, Rest: 10 sec (type: recovery)

    """


    for experiment_i in [1]:
        # A 1+1 strategy
        es = LLaMEA(
            f=evaluate_HIIT,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name="hiit1",
            elitism=True,
            HPO=False,
            budget=10,
        )
        print(es.run())
