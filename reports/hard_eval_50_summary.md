# Hard Evaluation 50 — Summary

**Date**: 2026-06-05 17:38:49
**API**: http://127.0.0.1:8000/predict

## Results

| Task | Pass | Total | Accuracy |
|------|------|-------|----------|
| Physics | 19 | 25 | 76.0% |
| Logic | 20 | 25 | 80.0% |
| **Total** | **39** | **50** | **78.0%** |

## Failure Modes by Tag

- **circuit** (2 fail): phys_03_voltage_divider, phys_04_current_divider
- **capacitor** (2 fail): phys_09_cap_disconnect, phys_10_cap_connected
- **dielectric** (2 fail): phys_09_cap_disconnect, phys_10_cap_connected
- **policy** (2 fail): logic_17_policy_all_required, logic_19_policy_violated
- **voltage_divider** (1 fail): phys_03_voltage_divider
- **current_divider** (1 fail): phys_04_current_divider
- **power** (1 fail): phys_05_power_dissipated
- **energy** (1 fail): phys_05_power_dissipated
- **unit_conversion** (1 fail): phys_05_power_dissipated
- **disconnected** (1 fail): phys_09_cap_disconnect
- **connected** (1 fail): phys_10_cap_connected
- **magnetic_field** (1 fail): phys_24_solenoid_B
- **solenoid** (1 fail): phys_24_solenoid_B
- **conjunction** (1 fail): logic_05_compound_missing
- **missing_condition** (1 fail): logic_05_compound_missing
- **double_negation** (1 fail): logic_12_double_negation
- **conjunction_n** (1 fail): logic_17_policy_all_required
- **violated_condition** (1 fail): logic_19_policy_violated
- **transitivity** (1 fail): logic_24_transitivity_long
- **deep_chain** (1 fail): logic_24_transitivity_long

## Individual Failures

### phys_03_voltage_divider (physics)
- **Question**: In a voltage divider with R1 = 4 kΩ and R2 = 6 kΩ in series across a 10 V source, what is the voltage across R2?
- **Expected**: `6 V`
- **Actual**: `10 V`
- **Tags**: ['circuit', 'voltage_divider']
- **Grade**: rel_err=0.667
- **Latency**: 6.87s
- **Explanation**: Used V = V_source. Extracted SI variables: V=10. Python computed the result as 10 V.

### phys_04_current_divider (physics)
- **Question**: A 12 mA current splits between two parallel branches with R1 = 100 Ω and R2 = 300 Ω. Find the current through R1.
- **Expected**: `9 mA`
- **Actual**: `0.009 A`
- **Tags**: ['circuit', 'current_divider']
- **Grade**: rel_err=0.999
- **Latency**: 18.3s
- **Explanation**: Direct LLM rescue solved this problem. Formula used: I_total = I_1 + I_2. The final result is 0.009 A. (unit verified)

### phys_05_power_dissipated (physics)
- **Question**: A 220 V appliance draws 5 A. What power does it dissipate in 30 minutes, in kJ?
- **Expected**: `1980 kJ`
- **Actual**: `1.1 kW`
- **Tags**: ['power', 'energy', 'unit_conversion']
- **Grade**: rel_err=0.999
- **Latency**: 6.85s
- **Explanation**: To find the power dissipated by the appliance, we use the formula P = V * I. Given that the voltage V is 220 V and the current I is 5 A, we compute P = 220 * 5 = 1100 W. Converting this to kilowatts, 

### phys_09_cap_disconnect (physics)
- **Question**: A 5 μF capacitor is charged to 100 V then disconnected. A dielectric of κ=4 is then inserted. What is the new voltage across the capacitor?
- **Expected**: `25 V`
- **Actual**: `Do not change`
- **Tags**: ['capacitor', 'dielectric', 'disconnected']
- **Grade**: no_number
- **Latency**: 6.57s
- **Explanation**: Search-backed qualitative registry reasoning: the parser did not need a numeric formula here, and the qualitative registry matched the question, so the answer is Do not change.

### phys_10_cap_connected (physics)
- **Question**: A 5 μF capacitor stays connected to a 100 V source. A dielectric of κ=4 is inserted. What is the new charge on the capacitor?
- **Expected**: `2 mC`
- **Actual**: `500 μC`
- **Tags**: ['capacitor', 'dielectric', 'connected']
- **Grade**: rel_err=249.000
- **Latency**: 7.24s
- **Explanation**: Used Q = C * V. Extracted SI variables: C=5e-06, V=100. Python computed the result as 0.0005 C.

### phys_24_solenoid_B (physics)
- **Question**: A solenoid has 1000 turns over a length of 50 cm and carries 2 A. Find the magnitude of the magnetic field inside.
- **Expected**: `5.03 mT`
- **Actual**: `unknown`
- **Tags**: ['magnetic_field', 'solenoid']
- **Grade**: actual_unknown
- **Latency**: 23.13s
- **Explanation**: The answer is unknown because the magnetic geometry is not fully specified for a safe deterministic formula.

### logic_05_compound_missing (logic)
- **Question**: Is Diego eligible for the scholarship?
- **Premises**: ['P1: A student is eligible for the scholarship if their GPA is above 3.5 and they submitted an application.', 'P2: Diego has a GPA of 3.8.']
- **Expected**: `unknown`
- **Actual**: `yes`
- **Tags**: ['conjunction', 'missing_condition']
- **Grade**: e=unknown != a=yes
- **Latency**: 10.66s
- **Explanation**: Answer is yes. Rule used: deterministic FOL->Z3 multi-hop entailment. Evidence: P1: A student is eligible for the scholarship if their GPA is above 3.5 and they submitted an application; P2: Diego has

### logic_12_double_negation (logic)
- **Question**: Is Lisa allowed to enter?
- **Premises**: ['P1: It is not the case that Lisa is not allowed to enter.']
- **Expected**: `yes`
- **Actual**: `no`
- **Tags**: ['double_negation']
- **Grade**: e=yes != a=no
- **Latency**: 8.2s
- **Explanation**: Answer is no. Rule used: explicit exception or negative fact overrides general rule. Evidence: P1: It is not the case that Lisa is not allowed to enter.

### logic_17_policy_all_required (logic)
- **Question**: Is Anna eligible for the certificate?
- **Premises**: ['P1: To get the certificate, a student must complete all assignments, pass the final exam, and attend at least 80% of classes.', 'P2: Anna completed all assignments.', 'P3: Anna passed the final exam.', 'P4: Anna attended 90% of classes.']
- **Expected**: `yes`
- **Actual**: `unknown`
- **Tags**: ['policy', 'conjunction_n']
- **Grade**: e=yes != a=unknown
- **Latency**: 12.29s
- **Explanation**: Answer is unknown. The selected premises are relevant, but they do not form a complete chain to the conclusion. Evidence: P1: To get the certificate, a student must complete all assignments, pass the 

### logic_19_policy_violated (logic)
- **Question**: Is Carla eligible for the certificate?
- **Premises**: ['P1: To get the certificate, a student must complete all assignments, pass the final exam, and attend at least 80% of classes.', 'P2: Carla completed all assignments.', 'P3: Carla did not pass the final exam.', 'P4: Carla attended 95% of classes.']
- **Expected**: `no`
- **Actual**: `unknown`
- **Tags**: ['policy', 'violated_condition']
- **Grade**: e=no != a=unknown
- **Latency**: 11.79s
- **Explanation**: Answer is unknown. The selected premises are relevant, but they do not form a complete chain to the conclusion. Evidence: P1: To get the certificate, a student must complete all assignments, pass the 

### logic_24_transitivity_long (logic)
- **Question**: Is Z a member of category X?
- **Premises**: ['P1: Every A is a B.', 'P2: Every B is a C.', 'P3: Every C is a D.', 'P4: Every D is an X.', 'P5: Z is an A.']
- **Expected**: `yes`
- **Actual**: `unknown`
- **Tags**: ['transitivity', 'deep_chain']
- **Grade**: e=yes != a=unknown
- **Latency**: 16.84s
- **Explanation**: Answer is unknown. The selected premises are relevant, but they do not form a complete chain to the conclusion. Evidence: P1: Every A is a B; P2: Every B is a C; P3: Every C is a D; P4: Every D is an 
