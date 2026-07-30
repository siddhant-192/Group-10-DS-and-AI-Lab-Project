# Zero-shot Text-to-SQL Error Analysis

Source results: `artifacts/zero-shot-eval/runs/20260720-012118/downloaded/results`

## Executive summary

The strongest baseline is **qwen3-4b-instruct-2507** at **72.34% execution accuracy**. Execution match is the primary metric; textual exact match understates semantically equivalent SQL.

| Tier | Model | Execution | Unique-SQL macro | Syntax valid | Format compliant | Failures |
|---|---|---|---|---|---|---|
| strong | qwen3-4b-instruct-2507 | 72.34% | 72.27% | 100.00% | 99.90% | 286 |
| middle | qwen2.5-coder-1.5b-instruct | 56.58% | 56.55% | 99.90% | 100.00% | 449 |
| weak-old | deepseek-coder-1.3b-instruct | 47.29% | 46.77% | 98.07% | 38.49% | 545 |

Key observations:

- The strongest model has 251 executable-but-wrong queries versus 35 database errors, so its main remaining problem is semantic construction rather than output validity.
- Its accuracy falls to **43.28%** on gold SQL longer than 30 tokens and **44.16%** on queries combining joins with subqueries.
- The nearly unchanged unique-SQL macro score (72.27%) shows that duplicated/paraphrased Spider queries are not materially inflating the headline score.
- Some failures expose annotation ambiguity: a model can follow the natural-language request more literally than the provided gold SQL. Those cases should be audited before using aggressive hard-example training.


## Accuracy by gold-query complexity

| Complexity | Support | qwen3-4b-instruct-2507 | qwen2.5-coder-1.5b-instruct | deepseek-coder-1.3b-instruct |
|---|---|---|---|---|
| simple | 593 | 82.12% | 66.78% | 61.38% |
| moderate | 336 | 61.91% | 50.59% | 31.25% |
| complex | 105 | 50.48% | 18.09% | 19.05% |

## Accuracy by SQL structure

These categories overlap; each row conditions on a feature in the gold query.

| Feature | Support | qwen3-4b-instruct-2507 | qwen2.5-coder-1.5b-instruct | deepseek-coder-1.3b-instruct |
|---|---|---|---|---|
| Aggregate | 551 | 69.87% | 62.07% | 45.73% |
| WHERE | 493 | 71.81% | 53.35% | 50.10% |
| Any join | 408 | 62.50% | 45.34% | 34.80% |
| 2+ joins | 88 | 53.41% | 19.32% | 19.32% |
| GROUP BY | 277 | 57.76% | 52.35% | 28.16% |
| HAVING | 79 | 62.02% | 49.37% | 32.91% |
| ORDER BY | 237 | 68.78% | 54.01% | 31.65% |
| LIMIT | 189 | 66.67% | 50.27% | 25.93% |
| DISTINCT | 87 | 59.77% | 43.68% | 39.08% |
| Subquery | 159 | 58.49% | 32.08% | 28.93% |
| Set operation | 80 | 57.50% | 23.75% | 23.75% |
| Join + subquery | 77 | 44.16% | 20.78% | 12.99% |
| Complex proxy | 105 | 50.48% | 18.09% | 19.05% |

## Failure composition

| Model | executable_wrong_result | database_execution_error | invalid_sql | non_sql_or_unsafe |
|---|---|---|---|---|
| qwen3-4b-instruct-2507 | 251 | 35 | 0 | 0 |
| qwen2.5-coder-1.5b-instruct | 218 | 230 | 1 | 0 |
| deepseek-coder-1.3b-instruct | 441 | 83 | 6 | 15 |

### Output-format behavior

| Model | Format category | Count |
|---|---|---|
| qwen3-4b-instruct-2507 | compliant | 1033 |
| qwen3-4b-instruct-2507 | trailing_text_or_multiple_statements | 1 |
| qwen2.5-coder-1.5b-instruct | compliant | 1034 |
| deepseek-coder-1.3b-instruct | markdown_fence | 617 |
| deepseek-coder-1.3b-instruct | compliant | 398 |
| deepseek-coder-1.3b-instruct | no_sql_in_response | 15 |
| deepseek-coder-1.3b-instruct | prose_before_sql | 3 |
| deepseek-coder-1.3b-instruct | trailing_text_or_multiple_statements | 1 |

### Most frequent structural mismatches among wrong parseable SQL

Mismatch labels are heuristic and non-mutually-exclusive; they localize likely error sources rather than prove causality.

| Model | Mismatch | Count |
|---|---|---|
| qwen3-4b-instruct-2507 | column_reference_mismatch | 191 |
| qwen3-4b-instruct-2507 | literal_or_value_mismatch | 99 |
| qwen3-4b-instruct-2507 | join_structure_mismatch | 99 |
| qwen3-4b-instruct-2507 | table_selection_mismatch | 88 |
| qwen3-4b-instruct-2507 | projection_count_mismatch | 86 |
| qwen3-4b-instruct-2507 | aggregation_mismatch | 47 |
| qwen3-4b-instruct-2507 | subquery_structure_mismatch | 46 |
| qwen3-4b-instruct-2507 | select_or_subquery_count_mismatch | 45 |
| qwen2.5-coder-1.5b-instruct | column_reference_mismatch | 347 |
| qwen2.5-coder-1.5b-instruct | join_structure_mismatch | 258 |
| qwen2.5-coder-1.5b-instruct | table_selection_mismatch | 253 |
| qwen2.5-coder-1.5b-instruct | literal_or_value_mismatch | 135 |
| qwen2.5-coder-1.5b-instruct | projection_count_mismatch | 113 |
| qwen2.5-coder-1.5b-instruct | select_or_subquery_count_mismatch | 84 |
| qwen2.5-coder-1.5b-instruct | subquery_structure_mismatch | 62 |
| qwen2.5-coder-1.5b-instruct | distinct_mismatch | 58 |
| deepseek-coder-1.3b-instruct | column_reference_mismatch | 405 |
| deepseek-coder-1.3b-instruct | projection_count_mismatch | 284 |
| deepseek-coder-1.3b-instruct | join_structure_mismatch | 250 |
| deepseek-coder-1.3b-instruct | table_selection_mismatch | 221 |
| deepseek-coder-1.3b-instruct | literal_or_value_mismatch | 174 |
| deepseek-coder-1.3b-instruct | select_or_subquery_count_mismatch | 151 |
| deepseek-coder-1.3b-instruct | subquery_structure_mismatch | 130 |
| deepseek-coder-1.3b-instruct | aggregation_mismatch | 101 |

### Database execution errors

| Model | Error | Count |
|---|---|---|
| qwen3-4b-instruct-2507 | unknown_column | 24 |
| qwen3-4b-instruct-2507 | sqlite_syntax_error | 5 |
| qwen3-4b-instruct-2507 | other_database_error | 4 |
| qwen3-4b-instruct-2507 | aggregate_misuse | 1 |
| qwen3-4b-instruct-2507 | unknown_table | 1 |
| qwen2.5-coder-1.5b-instruct | unknown_column | 212 |
| qwen2.5-coder-1.5b-instruct | unknown_table | 14 |
| qwen2.5-coder-1.5b-instruct | other_database_error | 3 |
| qwen2.5-coder-1.5b-instruct | aggregate_misuse | 2 |
| deepseek-coder-1.3b-instruct | unknown_column | 64 |
| deepseek-coder-1.3b-instruct | non_sql_or_unsafe | 15 |
| deepseek-coder-1.3b-instruct | ambiguous_column | 7 |
| deepseek-coder-1.3b-instruct | unknown_function | 6 |
| deepseek-coder-1.3b-instruct | sqlite_syntax_error | 4 |
| deepseek-coder-1.3b-instruct | other_database_error | 4 |
| deepseek-coder-1.3b-instruct | unknown_table | 3 |
| deepseek-coder-1.3b-instruct | aggregate_misuse | 1 |

## Difficulty slices for qwen3-4b-instruct-2507

| Dimension | Bin | Support | Execution |
|---|---|---|---|
| Gold SQL tokens | 01-10 | 373 | 83.65% |
| Gold SQL tokens | 11-20 | 383 | 72.32% |
| Gold SQL tokens | 21-30 | 211 | 61.61% |
| Gold SQL tokens | 31+ | 67 | 43.28% |
| Question words | 01-08 | 195 | 83.59% |
| Question words | 09-15 | 620 | 73.87% |
| Question words | 16+ | 219 | 57.99% |
| Schema tables | 01-03 | 571 | 74.43% |
| Schema tables | 04-06 | 303 | 69.31% |
| Schema tables | 07+ | 160 | 70.62% |
| Model input tokens | 001-256 | 316 | 87.97% |
| Model input tokens | 257-512 | 554 | 64.08% |
| Model input tokens | 513+ | 164 | 70.12% |

## Cross-model disagreement

- At least one model succeeds on **825/1034 (79.79%)** examples. This is the three-model oracle ceiling.
- All models succeed on **371 (35.88%)** examples.
- All models fail on **209 (20.21%)** examples.

| Correct model set | Examples |
|---|---|
| qwen3-4b-instruct-2507 + qwen2.5-coder-1.5b-instruct + deepseek-coder-1.3b-instruct | 371 |
| none | 209 |
| qwen3-4b-instruct-2507 + qwen2.5-coder-1.5b-instruct | 158 |
| qwen3-4b-instruct-2507 | 136 |
| qwen3-4b-instruct-2507 + deepseek-coder-1.3b-instruct | 83 |
| qwen2.5-coder-1.5b-instruct | 42 |
| deepseek-coder-1.3b-instruct | 21 |
| qwen2.5-coder-1.5b-instruct + deepseek-coder-1.3b-instruct | 14 |

## Hardest validation databases

### qwen3-4b-instruct-2507

| Database | Support | Execution |
|---|---|---|
| car_1 | 92 | 44.56% |
| wta_1 | 62 | 45.16% |
| world_1 | 120 | 57.50% |
| dog_kennels | 82 | 67.07% |
| tvshow | 62 | 67.74% |

### qwen2.5-coder-1.5b-instruct

| Database | Support | Execution |
|---|---|---|
| car_1 | 92 | 31.52% |
| world_1 | 120 | 35.83% |
| wta_1 | 62 | 38.71% |
| dog_kennels | 82 | 42.68% |
| student_transcripts_tracking | 78 | 43.59% |

### deepseek-coder-1.3b-instruct

| Database | Support | Execution |
|---|---|---|
| world_1 | 120 | 26.67% |
| car_1 | 92 | 30.43% |
| wta_1 | 62 | 38.71% |
| pets_1 | 42 | 40.48% |
| network_1 | 56 | 41.07% |

## Fine-tuning priorities

Priorities are ranked by the strongest model's observed failure volume and feature support. Sampling multipliers are caps, not additive weights, because examples belong to multiple categories.

| Priority | Feature | Val support | Strong failures | Strong accuracy | Train examples | Max multiplier |
|---|---|---|---|---|---|---|
| 1 | Aggregate | 551 | 166 | 69.87% | 3268 | 2.0x |
| 2 | Any join | 408 | 153 | 62.50% | 2770 | 2.0x |
| 3 | WHERE | 493 | 139 | 71.81% | 3502 | 1.5x |
| 4 | GROUP BY | 277 | 117 | 57.76% | 1773 | 2.5x |
| 5 | ORDER BY | 237 | 74 | 68.78% | 1625 | 2.0x |
| 6 | Subquery | 159 | 66 | 58.49% | 1017 | 2.5x |
| 7 | LIMIT | 189 | 63 | 66.67% | 1104 | 2.0x |
| 8 | Complex proxy | 105 | 52 | 50.48% | 760 | 2.5x |
| 9 | Join + subquery | 77 | 43 | 44.16% | 399 | 3.0x |
| 10 | 2+ joins | 88 | 41 | 53.41% | 826 | 2.5x |

Recommended training contract:

1. Train only on the 6,997 executable training examples; never mix validation databases into training.
2. Use feature-aware sampling with the maximum applicable multiplier per row, then cap repeated duplicates.
3. Keep the SQL-only assistant target and explicitly penalize prose or Markdown output.
4. Track execution accuracy overall and by feature; do not select checkpoints on text exact match alone.
5. Retain a natural-distribution validation score alongside the hard-feature slices so oversampling does not hide regressions.

## Representative qwen3-4b-instruct-2507 failures

### spider-validation-00066 — pets_1 / complex

Question: What is the first name of every student who has a dog but does not have a cat?

Issue: `column_reference_mismatch, literal_or_value_mismatch, join_structure_mismatch, projection_count_mismatch, subquery_structure_mismatch, distinct_mismatch`

Gold:

```sql
SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')
```

Prediction:

```sql
SELECT DISTINCT s.Fname
FROM "Student" s
JOIN "Has_Pet" hp ON s."StuID" = hp."StuID"
JOIN "Pets" p ON hp."PetID" = p."PetID"
WHERE p."PetType" = 'dog'
  AND NOT EXISTS (
    SELECT 1
    FROM "Has_Pet" hp2
    JOIN "Pets" p2 ON hp2."PetID" = p2."PetID"
    WHERE hp2."StuID" = s."StuID"
      AND p2."PetType" = 'cat'
  );
```

### spider-validation-00175 — car_1 / complex

Question: Which are the car makers which produce at least 2 models and more than 3 car makers ? List the id and the maker .

Issue: `table_selection_mismatch, join_structure_mismatch, projection_count_mismatch, subquery_structure_mismatch, set_operation_mismatch`

Gold:

```sql
select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >=  2 intersect select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model group by t1.id having count(*)  >  3;
```

Prediction:

```sql
SELECT cm.Id, cm.Maker
FROM car_makers cm
JOIN model_list ml ON cm.Id = ml.Maker
GROUP BY cm.Id, cm.Maker
HAVING COUNT(ml.Model) >= 2
AND cm.Id IN (
    SELECT cm2.Id
    FROM car_makers cm2
    JOIN model_list ml2 ON cm2.Id = ml2.Maker
    GROUP BY cm2.Id
    HAVING COUNT(ml2.Model) > 3
);
```

### spider-validation-00176 — car_1 / complex

Question: What are the ids and makers of all car makers that produce at least 2 models and make more than 3 cars?

Issue: `unknown_column`

Gold:

```sql
SELECT T1.Id ,  T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id HAVING count(*)  >=  2 INTERSECT SELECT T1.Id ,  T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model GROUP BY T1.Id HAVING count(*)  >  3;
```

Prediction:

```sql
SELECT cm.Id, cm.Maker
FROM car_makers cm
JOIN model_list ml ON cm.Id = ml.Maker
GROUP BY cm.Id, cm.Maker
HAVING COUNT(ml.Model) >= 2
AND COUNT(cars_data.Id) > 3;
```

### spider-validation-00700 — voter_1 / complex

Question: List the area codes in which voters voted both for the contestant 'Tabatha Gehling' and the contestant 'Kelly Clauss'.

Issue: `join_structure_mismatch, select_or_subquery_count_mismatch, projection_count_mismatch, subquery_structure_mismatch, set_operation_mismatch, distinct_mismatch`

Gold:

```sql
SELECT T3.area_code FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number  =  T2.contestant_number JOIN area_code_state AS T3 ON T2.state  =  T3.state WHERE T1.contestant_name  =  'Tabatha Gehling' INTERSECT SELECT T3.area_code FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number  =  T2.contestant_number JOIN area_code_state AS T3 ON T2.state  =  T3.state WHERE T1.contestant_name  =  'Kelly Clauss'
```

Prediction:

```sql
SELECT DISTINCT ac.area_code
FROM "AREA_CODE_STATE" ac
JOIN "VOTES" v ON ac.state = v.state
WHERE v.contestant_number IN (
    SELECT c.contestant_number
    FROM "CONTESTANTS" c
    WHERE c.contestant_name = 'Tabatha Gehling'
)
AND v.contestant_number IN (
    SELECT c.contestant_number
    FROM "CONTESTANTS" c
    WHERE c.contestant_name = 'Kelly Clauss'
);
```

### spider-validation-00505 — battle_death / complex

Question: Show names, results and bulgarian commanders of the battles with no ships lost in the 'English Channel'.

Issue: `unknown_column`

Gold:

```sql
SELECT name ,  RESULT ,  bulgarian_commander FROM battle EXCEPT SELECT T1.name ,  T1.result ,  T1.bulgarian_commander FROM battle AS T1 JOIN ship AS T2 ON T1.id  =  T2.lost_in_battle WHERE T2.location  =  'English Channel'
```

Prediction:

```sql
SELECT b.name, b.result, b.bulgarian_commander
FROM battle b
WHERE b.id NOT IN (
    SELECT s.lost_in_battle
    FROM ship s
    JOIN battle b ON s.lost_in_battle = b.id
    WHERE b.location = 'English Channel'
);
```

### spider-validation-00757 — world_1 / moderate

Question: What is the language that is used by the largest number of Asian nations?

Issue: `aggregate_misuse`

Gold:

```sql
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.Continent  =  "Asia" GROUP BY T2.Language ORDER BY COUNT (*) DESC LIMIT 1
```

Prediction:

```sql
SELECT cl.Language
FROM countrylanguage cl
JOIN country c ON cl.CountryCode = c.Code
WHERE c.Continent = 'Asia'
ORDER BY SUM(cl.Percentage) DESC
LIMIT 1;
```

### spider-validation-00177 — car_1 / complex

Question: What are the id and names of the countries which have more than 3 car makers or produce the 'fiat' model?

Issue: `table_selection_mismatch, literal_or_value_mismatch, join_structure_mismatch, select_or_subquery_count_mismatch, projection_count_mismatch, subquery_structure_mismatch, set_operation_mismatch, group_by_mismatch, having_mismatch`

Gold:

```sql
SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country GROUP BY T1.countryId HAVING count(*)  >  3 UNION SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country JOIN MODEL_LIST AS T3 ON T2.Id  =  T3.Maker WHERE T3.Model  =  'fiat';
```

Prediction:

```sql
SELECT 
    c."CountryId",
    c."CountryName"
FROM 
    "countries" c
JOIN 
    "car_makers" cm ON c."CountryName" = cm."Country"
WHERE 
    (SELECT COUNT(*) FROM "car_makers" WHERE "Country" = c."CountryName") > 3
    OR EXISTS (
        SELECT 1 
        FROM "model_list" ml 
        JOIN "car_names" cn ON ml."Model" = cn."Model" 
        WHERE ml."Maker" = (SELECT Id FROM "car_makers" WHERE "Maker" = 'fiat')
    );
```

### spider-validation-00178 — car_1 / complex

Question: What are the ids and names of all countries that either have more than 3 car makers or produce fiat model ?

Issue: `column_reference_mismatch, literal_or_value_mismatch, select_or_subquery_count_mismatch, projection_count_mismatch, subquery_structure_mismatch, distinct_mismatch`

Gold:

```sql
select t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  3 union select t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country join model_list as t3 on t2.id  =  t3.maker where t3.model  =  'fiat';
```

Prediction:

```sql
SELECT DISTINCT c."CountryId", c."CountryName"
FROM "countries" c
JOIN "car_makers" cm ON c."CountryName" = cm."Country"
WHERE c."Continent" IN (
    SELECT DISTINCT c2."Continent"
    FROM "countries" c2
    JOIN "car_makers" cm2 ON c2."CountryName" = cm2."Country"
    GROUP BY c2."Continent"
    HAVING COUNT(cm2."Id") > 3
)
UNION
SELECT DISTINCT c."CountryId", c."CountryName"
FROM "countries" c
JOIN "model_list" ml ON c."CountryName" = (SELECT cm."Country" FROM "car_makers" cm WHERE cm."Maker" = ml."Maker")
WHERE ml."Model" LIKE '%Fiat%';
```

## Metric caveats

This analysis uses the run's local read-only SQLite result-equivalence metric. It is stronger than string exact match for semantically equivalent SQL, but it is not the official Spider test-suite evaluator and can admit accidental matches on a particular database instance. The retained predictions can be scored with an official evaluator later without regenerating model outputs.

