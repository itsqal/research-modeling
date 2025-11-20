import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from fuzzy_config import SYSTEM_CONFIG

class RiskInferenceEngine:
    def __init__(self):
        self.config = SYSTEM_CONFIG
        self.sim = self._build_system()
        self.range_width = self.config['scaling']['max_val'] - self.config['scaling']['min_val']

    def _build_system(self):
        """Constructs the fuzzy inference system based on the configuration."""
        params = self.config['fuzzy_params']

        u_pattern = np.arange(0, 1.01, 0.01)
        u_province = np.arange(0, 2, 1)  
        u_how_long = np.arange(0, 51, 1)  
        u_age = np.arange(15, 76, 1)  
        u_how_often = np.arange(0, 31, 1) 
        u_risk = np.arange(0, 1.01, 0.01)

        pat = ctrl.Antecedent(u_pattern, 'pattern_match_score')
        prov = ctrl.Antecedent(u_province, 'province_aceh')
        long = ctrl.Antecedent(u_how_long, 'how_long_used_signs_years')
        age = ctrl.Antecedent(u_age, 'fisherman_age')
        often = ctrl.Antecedent(u_how_often, 'how_often_used_signs')

        risk = ctrl.Consequent(u_risk, 'risk')

        pat['Low'] = fuzz.trapmf(u_pattern, [0, 0, params['pattern']['slope_start'], params['pattern']['slope_end']])
        pat['High'] = fuzz.trapmf(u_pattern, [params['pattern']['slope_start'], params['pattern']['slope_end'], 1, 1])

        prov['Not_Aceh'] = fuzz.trimf(u_province, [0, 0, 0.1])
        prov['Is_Aceh'] = fuzz.trimf(u_province, [0.9, 1, 1])

        long['Short_Time'] = fuzz.trapmf(u_how_long, [0, 0, params['how_long']['slope_start'], params['how_long']['slope_end']])
        long['Long_Time'] = fuzz.trapmf(u_how_long, [params['how_long']['slope_start'], params['how_long']['slope_end'], 50, 50])

        age['Young'] = fuzz.trapmf(u_age, [15, 15, params['age']['slope_start'], params['age']['slope_end']])
        age['Mature'] = fuzz.trapmf(u_age, [params['age']['slope_start'], params['age']['slope_end'], 75, 75])

        often['Infrequent'] = fuzz.trapmf(u_how_often, [0, 0, params['how_often']['slope_start'], params['how_often']['slope_end']])
        often['Frequent'] = fuzz.trapmf(u_how_often, [params['how_often']['slope_start'], params['how_often']['slope_end'], 30, 30])

        # Output MFs (Wide Triangles for stability)
        risk['Moderate'] = fuzz.trimf(u_risk, [0, 0, 1])
        risk['High'] = fuzz.trimf(u_risk, [0, 1, 1])

        # Rules
        rules = [
            ctrl.Rule(pat['Low'] & prov['Not_Aceh'], risk['Moderate']),
            ctrl.Rule(pat['Low'] & prov['Is_Aceh'] & long['Short_Time'] & age['Young'], risk['High']),
            ctrl.Rule(pat['Low'] & prov['Is_Aceh'] & long['Short_Time'] & age['Mature'] & often['Infrequent'], risk['High']),
            ctrl.Rule(pat['Low'] & prov['Is_Aceh'] & long['Short_Time'] & age['Mature'] & often['Frequent'], risk['Moderate']),
            ctrl.Rule(pat['Low'] & prov['Is_Aceh'] & long['Long_Time'], risk['High']),
            ctrl.Rule(pat['High'], risk['High'])
        ]

        risk_ctrl = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(risk_ctrl)
    
    def predict(self, data: dict):
        """Main inference method to get risk score."""
        self.sim.input['pattern_match_score'] = data['weighted_sum_norm']
        self.sim.input['province_aceh'] = data['province']
        self.sim.input['how_long_used_signs_years'] = data['how_long_used_signs_years']
        self.sim.input['fisherman_age'] = data['fisherman_age']
        self.sim.input['how_often_used_signs'] = data['how_often_used_signs']

        try:
            self.sim.compute()
            raw_score = self.sim.output['risk']

            scaled = (raw_score - self.config['scaling']['min_val']) / self.range_width
            scaled = np.clip(scaled, 0.0, 1.0)

            label = 'MODERATE RISK'
            if scaled > 0.8: label = "HIGH RISK"

            return {
                "status": "success",
                "risk_score_raw": round(raw_score, 4),
                "risk_score_scaled": round(scaled, 4),
                "risk_label": label,
                "input_summary": {
                    "pattern_match_score": data['weighted_sum_norm'],
                    "province_aceh": "Aceh" if data['province'] == 1 else "D.I Yogyakarta",
                    "how_long_used_signs_years": data['how_long_used_signs_years'],
                    "fisherman_age": data['fisherman_age'],
                    "how_often_used_signs": data['how_often_used_signs'],
                }
            }
        except Exception as e:
            return { "status": "error", "message": str(e)}
