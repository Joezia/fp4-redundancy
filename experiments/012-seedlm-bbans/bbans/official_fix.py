"""The one patch applied to the official bits-back util (also used in B1 with --ppf-fix).

util.gaussian_latent_ppf returns the bucket index from norm.ppf + searchsorted only. When the floating-point cdf and
ppf disagree at a bucket edge, the check `start <= cf < start + freq` in util.non_uniform_dec_statfun fails.
Craystack (the authors' later library, craystack/codecs/codecs.py, _gaussian_ppf) adds a correction loop for this;
the same loop is added here. It moves the index only when cdf(idx) <= cf < cdf(idx + 1) does not hold, so every
pop that the official code completes gives the same result.
"""


def patch_ppf(util):
    orig = util.gaussian_latent_ppf

    def ppf_fixed(mean, stdd, prior_prec, post_prec):
        base = orig(mean, stdd, prior_prec, post_prec)
        cdf = util.gaussian_latent_cdf(mean, stdd, prior_prec, post_prec)
        top = (1 << prior_prec) - 1

        def ppf(cf):
            idx = int(base(cf))
            while not (cdf(idx) <= cf < cdf(idx + 1)):
                idx = idx - 1 if cf < cdf(idx) else idx + 1
                idx = min(max(idx, 0), top)
            return idx
        return ppf
    util.gaussian_latent_ppf = ppf_fixed
