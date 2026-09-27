# Full Ground-Truth Audit

- S1 rows: 2,206,821, S2 rows: 5,034,616, S3 rows: 5,285,603, ground-truth S1 rows: 2,206,821

## Audit A — Cross-country matches

- same-country positive links: 7,638,365 (100.00%)
- cross-country positive links: 0 (0.00%)
- missing/unknown country positive links: 0 (0.00%)

Cross-country examples (S1 id, S1 country, matched id, matched country):

Missing/unknown country examples (S1 id, S1 country, matched id, matched country):

**Action**: missing/unknown country is tracked separately from cross-country matches. If cross-country positives exist above zero, do NOT let blocking/normalization filter on country equality.

## Audit B — Numeric-overlap categories

- agreement: 4,315,021 (56.49%)
- partial: 1,333,481 (17.46%)
- none: 977,524 (12.80%)
- conflict: 1,012,339 (13.25%)

Examples — agreement:
  - ('S1-965667', 'S3-775321672', ['85'], ['85'])
  - ('S1-55344266', 'S2-249013014', ['29', '6', '2nd'], ['29', '6', '2nd'])
  - ('S1-55344266', 'S2-197070651', ['29', '6', '2nd'], ['29', '6', '2nd'])
  - ('S1-55344266', 'S3-478195123', ['29', '6', '2nd'], ['29', '6', '2nd'])
  - ('S1-55344266', 'S3-384364074', ['29', '6', '2nd'], ['29', '6', '2nd'])
  - ('S1-343815751', 'S3-878454467', ['630', '45th'], ['630', '45th'])
  - ('S1-102811957', 'S2-478959098', ['3315'], ['3315'])
  - ('S1-102811957', 'S2-553508714', ['3315'], ['3315'])
  - ('S1-102811957', 'S2-625774905', ['3315'], ['3315'])
  - ('S1-102811957', 'S3-728090388', ['3315'], ['3315'])
Examples — partial:
  - ('S1-343815751', 'S2-790675320', ['630', '45th'], ['630', '45nd'])
  - ('S1-656753428', 'S2-153058913', ['989', '9487203', '684'], ['989', '9487203', '0684'])
  - ('S1-656753428', 'S2-24659151', ['989', '9487203', '684'], ['9487203', '0684'])
  - ('S1-656753428', 'S3-679606215', ['989', '9487203', '684'], ['684'])
  - ('S1-318373630', 'S2-660036492', ['1', '3a'], ['1'])
  - ('S1-318373630', 'S3-804600254', ['1', '3a'], ['3a', '1', '316'])
  - ('S1-789009573', 'S2-383871912', ['14', '47', '13', '187c'], ['13', '187c'])
  - ('S1-789009573', 'S3-74481402', ['14', '47', '13', '187c'], ['517', '13', '187c'])
  - ('S1-789009573', 'S3-576451439', ['14', '47', '13', '187c'], ['517', '13', '187c'])
  - ('S1-503957000', 'S3-858763214', ['20085', '23'], ['23', '20085-20089'])
Examples — none:
  - ('S1-965667', 'S2-681193310', ['85'], [])
  - ('S1-965667', 'S2-743505751', ['85'], [])
  - ('S1-965667', 'S3-11291185', ['85'], [])
  - ('S1-965667', 'S3-860443364', ['85'], [])
  - ('S1-102811957', 'S3-449308785', ['3315'], [])
  - ('S1-29845983', 'S2-648035184', ['33'], [])
  - ('S1-29845983', 'S3-588502663', ['33'], [])
  - ('S1-730934468', 'S2-356983532', ['728'], [])
  - ('S1-274126313', 'S2-680265918', ['3907'], [])
  - ('S1-274126313', 'S3-925631694', ['3907'], [])
Examples — conflict:
  - ('S1-343815751', 'S2-479876582', ['630', '45th'], ['45nd'])
  - ('S1-18727616', 'S2-755677256', ['1056'], ['8807', '1056-1060'])
  - ('S1-18727616', 'S3-187831601', ['1056'], ['1056c'])
  - ('S1-18727616', 'S3-641489370', ['1056'], ['1056c'])
  - ('S1-18727616', 'S3-476250621', ['1056'], ['1056c'])
  - ('S1-546142636', 'S3-200008747', ['8706'], ['870'])
  - ('S1-546142636', 'S3-729771680', ['8706'], ['870'])
  - ('S1-727602285', 'S3-204655096', ['201/d'], ['01/d'])
  - ('S1-567308588', 'S2-191200521', ['9236'], ['9238'])
  - ('S1-567308588', 'S2-964733072', ['9236'], ['9238'])

**Action**: hand these category counts to Parimarjan for numeric-feature design. Do not turn a sample observation into an unconditional hard rule.

## Audit C — Blank-address behavior among TRUE matches

- positive links where S1 or matched address is blank: 337,018 / 7,638,365 (4.41%)

Examples (s1_id, s1_name, matched_id, matched_name):
  - ('S1-965667', 'Maure Williams Colombier Inc', 'S2-681193310', 'Maure Wilblims Colombier Inc')
  - ('S1-965667', 'Maure Williams Colombier Inc', 'S2-743505751', 'Maure Williams Colombier')
  - ('S1-965667', 'Maure Williams Colombier Inc', 'S3-860443364', 'Maure Williams Inc Center')
  - ('S1-29845983', 'Hendricks and Flowers Inc', 'S3-588502663', 'Hendricks and Inc Flowers')
  - ('S1-730934468', 'Orellana Investments LLC', 'S2-356983532', 'Orellana Investments Investments Llc')
  - ('S1-274126313', 'Obsidian, LLC', 'S2-680265918', 'obsidian, llc')
  - ('S1-274126313', 'Obsidian, LLC', 'S3-461175723', 'Obsidian, [[LLC]]')
  - ('S1-145361722', 'Dick Regional Armada Corp', 'S2-120366543', 'Dick Regional')
  - ('S1-65263544', 'Caldeon Nova', 'S2-998270769', 'CALDEON NOVA LLC')
  - ('S1-840162906', 'Summit Health LLC', 'S2-129529678', 'Summit Health Enterprises')
  - ('S1-9962387', 'Systel Buildstructure (India) Private Limited', 'S2-15221089', 'Systel Buildstructure (India)')
  - ('S1-116043204', 'Red Consultants Pvt. Ltd.', 'S3-85523430', 'Red Pvt. Ltd. Center')
  - ('S1-876102895', 'Physical Therapy Clinic', 'S2-791711694', 'Physical [Clinic] Therapy')
  - ('S1-67172650', 'Apar Engineering (India) Private Limited', 'S3-251407198', 'M/s Apar Engineering (India)')
  - ('S1-131928575', 'Jeanlouis Advantage LLC', 'S3-504772212', 'Jeanlouis Advantage Llc')

**Action**: make sure missingness is represented explicitly so a blank address does not create a false similarity or a false non-match.

## Audit D — Match-count distribution

- 0 match(es): 123,247 S1 entities (5.58%)
- 1 match(es): 119,157 S1 entities (5.40%)
- 2 match(es): 375,212 S1 entities (17.00%)
- 3 match(es): 530,841 S1 entities (24.05%)
- 4 match(es): 484,115 S1 entities (21.94%)
- 5 match(es): 321,957 S1 entities (14.59%)
- 6 match(es): 164,868 S1 entities (7.47%)
- 7 match(es): 63,968 S1 entities (2.90%)
- 8 match(es): 18,680 S1 entities (0.85%)
- 9 match(es): 4,205 S1 entities (0.19%)
- 10 match(es): 534 S1 entities (0.02%)
- 11 match(es): 37 S1 entities (0.00%)

**Action**: confirm the decision layer supports zero, one, or many matches per S1 entity — do not assume a fixed match count.
