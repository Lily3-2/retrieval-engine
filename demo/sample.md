# Employee Leave Policy Specification

## 1. Overview

This document defines the accrual and carryover rules for annual leave.
See Table 1 for the accrual rate schedule and Section 2.1 for proration rules.

## 2. Accrual Rules

### 2.1 Proration

Proration applies to employees who join mid-year. As per Section 3, the
proration factor is computed monthly.

New employees accrue leave starting from their date of joining, prorated
by the number of complete months remaining in the calendar year.

### 2.2 Ceiling

The maximum accrual ceiling is 42 days. Any balance above this ceiling is
forfeited at year-end unless covered by an exception in Appendix A.

Accrual rules by grade:

- Grade A: 1.75 days per month
- Grade B: 1.50 days per month
- Grade C: 1.25 days per month

## 3. Carryover

Carryover is calculated at fiscal year-end. See Table 1 for the full
accrual schedule referenced above.

| Grade | Monthly Accrual | Annual Cap |
|-------|-----------------|------------|
| A     | 1.75            | 21.0       |
| B     | 1.50            | 18.0       |
| C     | 1.25            | 15.0       |

## Appendix A: Exceptions

> Employees on approved sabbatical retain their full accrual ceiling
> regardless of the standard forfeiture rule in Section 2.2.

```python
def prorate(months_remaining, monthly_rate):
    return round(months_remaining * monthly_rate, 2)
```
