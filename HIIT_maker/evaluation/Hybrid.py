import json
from evaluation.ABtest import ABtest
from evaluation.LLMpredict import LLMpredict
import utils.state as state
from utils.json_tools import _coerce_json


class Hybrid:
    """Hybrid evaluator combining LLM predictions and AB testing."""
    def __init__(self, llm, llm_eval, logger=None, user_interval = 5):
        self.llm = llm
        self.llm_eval = llm_eval
        self.logger = logger
        self.user_interval = user_interval

        self.ab_test = ABtest(logger=logger)
        self.llm_predict= LLMpredict(llm, logger=logger)

        self.iteration = 0

    def evaluate_HIIT(self, solution, logger=None, llm_eval= None):
        self.iteration += 1
        S = state._CHOICE_STATE

        # Decide which evaluator to use
        if self.iteration % self.user_interval == 0:
            # Use ABtest (user choose)
            if self.logger:
                self.logger.info(f"Hybrid] Iteration {self.iteration}: ABtest evaluation.")

            prev_incumbent = S["incumbent_id"]
            solution = self.ab_test.evaluate_HIIT(solution, logger=self.logger or logger)
            S = state._CHOICE_STATE
            if S["incumbent_id"] != prev_incumbent:
                program = []
                try:
                    data = S["incumbent_json"]
                    if data:
                        data = _coerce_json(data)
                        if not isinstance(data, dict):
                            raise ValueError(f"❌ data is not a dict but {type(data)}")
                        
                        S["incumbent_json"] = data  # update global with parsed version

                        # Collect exercises from JSON structure
                        if "warm_up" in data:
                            program += data["warm_up"]
                        if "main_set" in data and "exercises" in data["main_set"]:
                            program += data["main_set"]["exercises"]
                        if "cool_down" in data:
                            program += data["cool_down"]

                        if not program:
                            raise ValueError("No exercises found in JSON input.")

                        if llm_eval is None:
                            try: 
                                llm_eval = globals().get("llm")
                            except KeyError:
                                raise RuntimeError("llm_eval is None and no global 'llm' found.")

                        # Step 1: Build prompt and predict
                        program_text = self.llm_predict._render_program_text(program)
                        prompt = self.llm_predict.make_prompt(program_text)
                        predictions = self.llm_predict.predict_hr_power(self.llm_eval, prompt)

                        if not predictions or not isinstance(predictions, dict):
                            raise ValueError("❌ LLM returned None instead of predictions.")
                        
                        # Step 2: Save predictions into JSON
                        data["predicted_hr"] = {
                            "summary": predictions.get("summary", {}),
                            "per_interval_hr": predictions.get("per_interval_hr", [])
                        }
                        data["predicted_power"] = {
                            "per_interval_power": predictions.get("per_interval_power", [])
                        }
                        if data.get("predicted_hr") is None or data.get("predicted_power") is None:
                            raise ValueError(f"❌ Missing prediction keys in data: "
                                            f"predicted_hr={type(data.get('predicted_hr'))}, "
                                            f"predicted_power={type(data.get('predicted_power'))}")

                        # Step 3: Compute fitness score using the same LLMpredict method
                        hr_max = self.llm_predict._infer_hr_max(predictions, default=200)
                        fitness_score = self.llm_predict.fitness(data, hr_max)
                        data["fitness"] = fitness_score

                        # Step 4: Update global state
                        S["incumbent_json"] = data
                        S["fitness_score"] = fitness_score

                        solution.set_scores(
                            fitness=S["fitness_score"],
                            feedback=f"Hybrid evaluation — model fitness: {S['fitness_score']:.3f}"
                        )

                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"[Hybrid] Failed to compute fitness for user-chosen B: {e}")

        else: 
            # Use LLMpredict (automatic prediction)
            if self.logger:
                self.logger.info(f"Hybrid] Iteration {self.iteration}: LLMpredict evaluation.")
            solution = self.llm_predict.evaluate_HIIT(solution, logger=self.logger or logger, llm_eval=llm_eval)  

        return solution

        
    
    