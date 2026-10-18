"""Exact random-audit stopping rules. Plotting and manuscript edits are separate."""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from scipy.special import gammaln
from scipy.stats import hypergeom, binom


def all_targets_in_prefix(referred, targets, searched):
    """Targets are uniformly positioned within the random referral order."""
    t, m, b = int(referred), int(targets), int(searched)
    if m == 0:
        return 1.0
    if b < m or t < m:
        return 0.0
    b = min(t, b)
    return float(np.exp(gammaln(b+1)-gammaln(b-m+1)-gammaln(t+1)+gammaln(t-m+1)))


def stopped_rule(pool, targets, empty_referrals, allowance, review_limit=None):
    """Fixed potential referrals; random order; stop at the allowance or read limit.

    Every target is assumed referable in this conditional calculation.
    Returns pre-physical-detection target success, recovered target count,
    required searches, and image readings.
    """
    N, m, k, b = map(int, [pool, targets, empty_referrals, allowance])
    n = N if review_limit is None else int(review_limit)
    assert 0 <= m <= N and 0 <= k <= N-m and b >= 0 and 0 <= n <= N
    t = m+k
    if b == 0:
        return dict(success=float(m == 0), captured=0., searches=0., readings=0.)
    if t == 0:
        return dict(success=1., captured=0., searches=0., readings=float(n))
    if n == N:
        searched = min(t, b)
        readings = N if b > t else b*(N+1)/(t+1)
        return dict(success=all_targets_in_prefix(t, m, searched),
                    captured=m*searched/t, searches=float(searched), readings=float(readings))
    # The number of potential referrals in the first n random positions.
    counts = np.arange(max(0, n-(N-t)), min(n, t)+1)
    weights = hypergeom.pmf(counts, N, t, n)
    searched = np.minimum(counts, b)
    success = np.array([all_targets_in_prefix(t, m, v) for v in searched])
    # A next image is read exactly when fewer than b referrals preceded it.
    readings = float(hypergeom.cdf(b-1, N, t, np.arange(n)).sum())
    return dict(success=float(weights@success), captured=float(m*(weights@searched)/t),
                searches=float(weights@searched), readings=readings)


def fixed_rule(pool, targets, empty_referrals, allowance, reviews):
    N, m, k, b, n = map(int, [pool, targets, empty_referrals, allowance, reviews])
    if n < m or b < m:
        success = 0.
    else:
        inclusion = float(hypergeom.pmf(m, N, m, n)) if N else 1.
        work = float(hypergeom.cdf(b-m, N-m, k, n-m)) if N > m else 1.
        success = inclusion*work
    return dict(success=success, captured=m*n/N if N else 0.,
                searches=(m+k)*n/N if N else 0., readings=float(n))


def stopped_with_referral_retention(pool, targets, empty_referrals, allowance, retention=1.):
    """Independent positive referral loss; empty potential responses remain fixed."""
    N, m, k, b = map(int, [pool, targets, empty_referrals, allowance])
    answer = dict(success=retention**m*stopped_rule(N, m, k, b)['success'],
                  captured=0., searches=0., readings=0.)
    for h in range(m+1):
        p = float(binom.pmf(h, m, retention))
        result = stopped_rule(N, h, k, b)
        for key in ['captured', 'searches', 'readings']:
            answer[key] += p*result[key]
    return answer


def replay_order(order, target, referral, allowance, read_limit=None):
    """Observable control: only preceding referrals and remaining slots are used."""
    n = len(order) if read_limit is None else read_limit
    searches = readings = captured = 0
    for i in order:
        if readings >= n or searches >= allowance:
            break
        readings += 1
        if referral[i]:
            searches += 1
            captured += int(target[i])
    return dict(success=float(captured == int(np.sum(target))), captured=float(captured),
                searches=float(searches), readings=float(readings))


def optimal_binary_history(pool, targets, empty_referrals, allowance):
    """Bellman enumeration of all stop/continue decisions using binary history.

    Referral types are exchangeable and not revealed by physical-search outcomes.
    A state records only readings and referrals. The stop reward is the chance
    that all targets are already among the referred bags.
    """
    N,m,k,z=map(int,[pool,targets,empty_referrals,allowance]);t=m+k
    @lru_cache(None)
    def value(read, found):
        stop=all_targets_in_prefix(t,m,found)
        if read==N or found>=z:
            return stop,0.
        p=(t-found)/(N-read)
        yes=value(read+1,found+1) if p>0 else (0.,0.)
        no=value(read+1,found) if p<1 else (0.,0.)
        continued=p*yes[0]+(1-p)*no[0]
        readings=1+p*yes[1]+(1-p)*no[1]
        # With equal reliability choose the action using fewer readings.
        if stop>=continued-1e-13:
            return stop,0.
        return continued,readings
    result=value(0,0)
    return dict(success=result[0],readings=result[1],states=value.cache_info().currsize)
