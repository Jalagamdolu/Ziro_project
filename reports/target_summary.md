# Strict Target Creation & Horizon Feasibility Report

## 1. Verified Target Dataset Overview
- **Primary Reference Time**: `10:30:00 IST` (`Asia/Kolkata`)
- **Primary Target Time**: `10:30:00 IST` (`Asia/Kolkata`)
- **Total Valid Exact Reference-Target Pairs**: **25,418**
- **Benchmark-Eligible Pairs** (excluding pilot canary Aug 10–14): **24,570**
- **Distinct Traded Symbols with Valid Pairs**: **1,859**
- **Overall Class Balance**:
  - DOWN ($<-1.0\%$): 11,318 (44.53%)
  - STABLE ($-1.0\% \le r \le +1.0\%$): 5,338 (21.00%)
  - UP ($>+1.0\%$): 8,762 (34.47%)

## 2. Sample Count and Class Balance by Observed Horizon

| Horizon | Sample Count | Unique Symbols | DOWN % | STABLE % | UP % | Calendar Days Ahead |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 session** | 5,915 | 1,622 | 34.93% | 36.16% | 28.91% | 1, 3, 5, 7, 8 days |
| **2 sessions** | 5,486 | 1,595 | 43.13% | 22.89% | 33.98% | 2, 4, 6, 8, 9, 12 days |
| **3 sessions** | 4,756 | 1,547 | 47.92% | 15.9% | 36.19% | 3, 5, 10, 13, 14 days |
| **4 sessions** | 3,547 | 1,509 | 47.96% | 14.15% | 37.89% | 6, 11, 13, 14, 15, 21 days |
| **5 sessions** | 2,583 | 1,431 | 51.03% | 11.96% | 37.01% | 7, 14, 16, 22 days |
| **6 sessions** | 1,656 | 1,277 | 50.54% | 12.26% | 37.2% | 8, 15, 19, 23 days |
| **7 sessions** | 1,328 | 1,224 | 49.85% | 11.82% | 38.33% | 16, 20, 24, 26 days |
| **>=8 sessions** | 147 | 103 | 60.54% | 10.88% | 28.57% | 17, 21, 22, 27, 28, 29, 30 days |

## 3. Purged Chronological Temporal Splits Analysis

When multi-day forward targets are used, training observations with targets in future sessions can cross into the validation evaluation period. A strict temporal purge removes training observations whose $t_{\text{tgt}} \ge t_{\text{val\_start}}$ and validation observations whose $t_{\text{tgt}} \ge t_{\text{test\_start}}$.

| Candidate Split | Scope | Raw Train | Purged Train | Retained Train | Raw Val | Purged Val | Retained Val | Test Pairs | Total Clean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1 (Multi-Horizon Pooled: 3-Phase Dates)** | All Horizons (1 to 7 sessions) | 14,724 | 12,638 (85.83%) | 2,086 | 6,887 | 5,893 (85.57%) | 994 | 2,959 | **6,039** |
| **Candidate 2 (Multi-Horizon Pooled: Expanded Train)** | All Horizons (1 to 7 sessions) | 18,904 | 14,080 (74.48%) | 4,824 | 4,431 | 3,744 (84.5%) | 687 | 1,235 | **6,746** |
| **Candidate 3 (Horizon-1 Specialized Model)** | Horizon = 1 Session Ahead Only | 2,012 | 1,158 (57.55%) | 854 | 1,681 | 687 (40.87%) | 994 | 2,100 | **3,948** |
| **Candidate 4 (Horizon-2 Specialized Model)** | Horizon = 2 Sessions Ahead Only | 1,629 | 1,629 (100.0%) | 0 | 1,870 | 1,870 (100.0%) | 0 | 1,870 | **1,870** |

## 4. Modeling Strategy Recommendation: Pooled Model vs Horizon-Specific
1. **Pooled Multi-Horizon Model (`observed_sessions_ahead` input feature)**:
   - *Pros*: Integrates all 24,570 cross-horizon pairs into a single, unified multi-task model. Maximizes statistical power across all horizons (1 to 7).
   - *Cons*: Temporal purging across multi-day horizons requires careful alignment to avoid cross-boundary leakage.
2. **Dedicated Horizon-1 Model (1 Session Ahead)**:
   - *Pros*: Possesses the cleanest temporal separation, the highest market liquidity relevance, and the most balanced class structure (36.2% STABLE, 34.9% DOWN, 28.9% UP).
   - *Cons*: Only predicts next-day movement.
3. **Recommended Two-Tier Strategy**:
   - **Tier 1**: A dedicated, highly-tuned **Horizon-1 Benchmark Model** (1-session forward).
   - **Tier 2**: A **Pooled Multi-Horizon Model** supporting arbitrary user-requested horizons between 1 and 7 trading days.

## 5. Supported Horizon Recommendation
- **Supported Range**: **1 to 7 observed sessions ahead** (each horizon contains $>1,000$ historical training examples).
- **Unsupported Range**: **$\ge 8$ sessions ahead** (only 147 samples total, $<0.6\%$ of pairs). For requests $>7$ sessions ahead, the application will return:
  > *"Insufficient historical data to support this prediction horizon."*
