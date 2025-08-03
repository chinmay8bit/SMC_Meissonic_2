# Using Meissonic's scheduler
## Best-of-N
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**best_reward**: [6.961170673370361, 6.907285690307617, 6.878042697906494, 7.017531394958496, 7.002593040466309, 7.006130218505859, 6.973943710327148, 6.904730796813965, 6.965787887573242, 7.1022515296936035, 7.002797603607178, 6.799677848815918]  
**Average**: 6.96  (12)

---

**CFG**: 5  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**best_reward**: [6.8366498947143555, 6.6916046142578125, 6.871970176696777]  
**Average**: 6.80  (3)

---

## SMC with Reverse as proposal
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 20  
**partial_resampling**: False  
**best_reward**: [6.7492170333862305, 6.821385860443115, 6.7521562576293945]  
**Average**: 6.77  (3)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 3  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: False  
**best_reward**: [6.300812721252441, 6.637012958526611, 6.870999336242676]  
**Average**: 6.60  (3)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 0.1  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: False  
**best_reward**: [6.718665599822998, 6.790803909301758, 6.78852653503418]  
**Average**: 6.77  (3)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: False  
**lambda_one_at**: None  
**resample_frequency**: 10  
**partial_resampling**: False  
**best_reward**: [6.609953880310059, 6.898380279541016, 6.75998592376709]  
**Average**: 6.76  (3)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: False  
**best_reward**: [7.004961967468262, 6.630211353302002, 6.826918601989746]  
**Average**: 6.82  (3)

---

## SMC with Locally Optimal Proposal
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [7.187208652496338, 7.17878532409668, 7.158297538757324, 7.107602119445801, 7.368976593017578, 7.185915946960449, 6.917961120605469, 7.225987911224365, 6.998950958251953, 7.204327583312988]  
**Average**: 7.15  (10)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 3  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [7.140955924987793, 7.413941383361816, 7.130088806152344, 7.369669437408447, 7.149348258972168, 7.201818466186523, 7.420206546783447, 7.0563249588012695]  
**Average**: 7.24  (8)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 3  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 30  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: False  
**best_reward**: [6.91611385345459, 6.907509803771973, 6.801436424255371, 7.034332752227783, 6.986466407775879]  
**Average**: 6.93  (5)

---

**CFG**: 5  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [7.076106071472168, 7.170135498046875, 6.949668884277344, 7.062686920166016]  
**Average**: 7.06  (4)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: False  
**best_reward**: [6.749275207519531, 6.727602005004883, 6.914567947387695]  
**Average**: 6.80  (3)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: False  
**lambda_one_at**: None  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: False  
**best_reward**: [6.836775779724121, 6.8625807762146, 7.054157257080078, 6.97995662689209, 7.014120578765869]  
**Average**: 6.95  (5)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 0.1  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [7.368734836578369, 6.967567443847656, 7.148920059204102, 7.428163051605225, 6.966780662536621, 7.103764533996582]  
**Average**: 7.16  (6)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 30  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: False  
**best_reward**: [6.966841697692871, 6.8676042556762695, 7.08735466003418, 6.92525053024292, 7.040090560913086]  
**Average**: 6.98  (5)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 0.1  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 30  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: False  
**best_reward**: [7.177652359008789, 7.092243194580078, 6.8036627769470215, 6.644637107849121, 6.842200756072998, 6.953699588775635]  
**Average**: 6.92  (6)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 30  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [6.5329976081848145, 6.8622331619262695]  
**Average**: 6.70  (2)

---

**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: False  
**lambda_one_at**: None  
**resample_frequency**: 10  
**partial_resampling**: False  
**continuous_formulation**: False  
**best_reward**: [7.074452877044678, 6.999091148376465, 6.8048810958862305, 7.026575088500977, 6.793296813964844]  
**Average**: 6.94  (5)

---

## SMC with Locally Optimal Proposal - Straight through grads
# Using ReMDM scheduler
## Best-of-N
**remdm_schedule**: cosine  
**remdm_remask_strategy**: rescale  
**remdm_eta**: 0.05  
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**best_reward**: [6.819243431091309, 6.816343307495117, 6.956999778747559]  
**Average**: 6.86  (3)

---

**remdm_schedule**: cosine  
**remdm_remask_strategy**: rescale  
**remdm_eta**: 0.1  
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**best_reward**: [6.944384574890137, 6.836662292480469, 6.784484386444092, 6.79617977142334]  
**Average**: 6.84  (4)

---

## SMC with Reverse as proposal
## SMC with Locally Optimal Proposal
**remdm_schedule**: cosine  
**remdm_remask_strategy**: rescale  
**remdm_eta**: 0.05  
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.05  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [6.890366077423096, 7.239747047424316, 6.890227317810059, 6.787561416625977]  
**Average**: 6.95  (4)

---

**remdm_schedule**: cosine  
**remdm_remask_strategy**: rescale  
**remdm_eta**: 0.05  
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.01  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [6.221607685089111, 6.200217247009277]  
**Average**: 6.21  (2)

---

**remdm_schedule**: cosine  
**remdm_remask_strategy**: rescale  
**remdm_eta**: 0.1  
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 1  
**tau**: 1.0  
**kl_weight**: 0.05  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [6.890666961669922, 6.940825462341309, 6.915220260620117, 6.861484050750732]  
**Average**: 6.90  (4)

---

**remdm_schedule**: cosine  
**remdm_remask_strategy**: rescale  
**remdm_eta**: 0.05  
**CFG**: 9  
**steps**: 100  
**reward_name**: aesthetic  
**prompt**: A large body of water with a rock in the middle and mountains in the background.  
**num_images**: 8  
**phi**: 3  
**tau**: 1.0  
**kl_weight**: 0.05  
**lambda_tempering**: True  
**lambda_one_at**: 100  
**resample_frequency**: 10  
**partial_resampling**: True  
**continuous_formulation**: True  
**best_reward**: [6.913580894470215, 6.773632526397705]  
**Average**: 6.84  (2)

---

## SMC with Locally Optimal Proposal - Straight through grads
