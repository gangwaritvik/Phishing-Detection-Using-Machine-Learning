import re
import math
from urllib.parse import urlparse, parse_qs
from collections import Counter


class URLFeatureExtractor:
    """Extracts features from URLs aligned with the PhiUSIIL dataset."""

    SUSPICIOUS_TLDS = {
        '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.club',
        '.online', '.site', '.website', '.space', '.live', '.stream',
        '.download', '.click', '.link', '.bid', '.loan', '.win'
    }

    BRAND_KEYWORDS = [
        'paypal', 'google', 'facebook', 'amazon', 'apple', 'microsoft',
        'netflix', 'instagram', 'twitter', 'linkedin', 'ebay', 'bank',
        'secure', 'account', 'update', 'verify', 'login', 'signin',
        'password', 'billing', 'payment', 'confirm'
    ]
    
    SUSPICIOUS_WORDS = [
    "login","verify","account","secure","update","password",
    "confirm","bank","billing","payment","signin","reset"
    ]

    SHORTENING_SERVICES = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'tiny.cc', 'lnkd.in', 'db.tt', 'qr.ae',
        'rebrand.ly', 'cutt.ly', 'shorturl.at', 'clck.ru'
    ]

    def __init__(self):
        self.features = {}

    def extract(self, url: str) -> dict:
        """Extract all features from a URL. Returns dict of feature_name -> value."""
        self.features = {}
        self._url = url.strip()

        try:
            parsed = urlparse(self._url if '://' in self._url else 'http://' + self._url)
        except Exception:
            parsed = urlparse('http://invalid.url')

        self._parsed = parsed
        domain = parsed.netloc or parsed.path.split('/')[0]
        domain = re.sub(r'^www\.', '', domain.lower())
        self._domain = domain
        path = parsed.path or ''
        query = parsed.query or ''
        full_url = self._url

        # ── 1. URL-level features ────────────────────────────────────────────
        self._add('URLLength',            len(full_url))
        self._add('DomainLength',         len(domain))
        self._add('IsDomainIP',           1 if self._is_ip(domain) else 0)
        self._add('URLSimilarityIndex',   self._url_similarity_index(full_url))
        self._add('CharContinuationRate', self._char_continuation_rate(full_url))
        self._add('TLDLegitimateProb',    self._tld_legit_prob(domain))
        self._add('URLCharProb',          self._url_char_prob(full_url))
        self._add('TLDLength',            len(self._get_tld(domain)))
        self._add('NoOfSubDomain',        self._count_subdomains(domain))
        self._add('HasObfuscation',       1 if self._has_obfuscation(full_url) else 0)
        self._add('NoOfObfuscatedChar',   self._count_obfuscated(full_url))
        self._add('ObfuscationRatio',     self._obfuscation_ratio(full_url))

        # ── 2. Special-character counts ──────────────────────────────────────
        self._add('NoOfLettersInURL',     sum(c.isalpha() for c in full_url))
        self._add('LetterRatioInURL',     sum(c.isalpha() for c in full_url) / max(len(full_url), 1))
        self._add('NoOfDegitsInURL',      sum(c.isdigit() for c in full_url))
        self._add('DegitRatioInURL',      sum(c.isdigit() for c in full_url) / max(len(full_url), 1))
        self._add('NoOfEqualsInURL',      full_url.count('='))
        self._add('NoOfQMarkInURL',       full_url.count('?'))
        self._add('NoOfAmpersandInURL',   full_url.count('&'))
        self._add('NoOfOtherSpecialCharsInURL', self._count_special(full_url))
        self._add('SpacialCharRatioInURL', self._special_char_ratio(full_url))

        # ── 3. HTTPS / protocol ──────────────────────────────────────────────
        self._add('IsHTTPS',              1 if parsed.scheme == 'https' else 0)

        # ── 4. Path & query features ─────────────────────────────────────────
        self._add('LineOfCode',           path.count('/'))
        self._add('LargestLineLength',    max((len(p) for p in path.split('/') if p), default=0))
        self._add('HasTitle',             1 if re.search(r'<title', full_url, re.I) else 0)
        self._add('DomainTitleMatchScore',self._domain_title_score(domain, full_url))
        self._add('URLTitleMatchScore',   self._url_title_score(full_url))
        self._add('HasFavicon',           0)
        self._add('Robots',               0)
        self._add('IsResponsive',         0)
        self._add('NoOfURLRedirect',      full_url.count('http', 1))
        self._add('NoOfSelfRedirect',     0)
        self._add('HasDescription',       0)
        self._add('NoOfPopup',            0)
        self._add('NoOfiFrame',           0)
        self._add('HasExternalFormSubmit',0)
        self._add('HasSocialNet',         1 if any(s in full_url.lower() for s in ['facebook','twitter','instagram','linkedin']) else 0)
        self._add('HasSubmitButton',      0)
        self._add('HasHiddenFields',      0)
        self._add('HasPasswordField',     1 if 'password' in full_url.lower() else 0)
        self._add('Bank',                 1 if 'bank' in full_url.lower() else 0)
        self._add('Pay',                  1 if any(w in full_url.lower() for w in ['pay','payment','paypal']) else 0)
        self._add('Crypto',               1 if any(w in full_url.lower() for w in ['crypto','bitcoin','wallet','coin']) else 0)

        # ── 5. Domain-based ──────────────────────────────────────────────────
        self._add('HasCopyrightInfo',     0)
        self._add('NoOfImage',            0)
        self._add('NoOfCSS',              0)
        self._add('NoOfJS',               0)
        self._add('NoOfSelfRef',          0)
        self._add('NoOfEmptyRef',         0)
        self._add('NoOfExternalRef',      0)

        # ── 6. Extra phishing signals ────────────────────────────────────────
        self._add('HasIPAddress',         1 if self._is_ip(domain) else 0)
        self._add('HasAtSymbol',          1 if '@' in full_url else 0)
        self._add('PrefixSuffixInDomain', 1 if '-' in domain else 0)
        self._add('SubDomainCount',       self._count_subdomains(domain))
        self._add('SSLState',             1 if parsed.scheme == 'https' else 0)
        self._add('DomainAge',            -1)  # requires WHOIS; -1 = unknown
        self._add('IsShortenedURL',       1 if self._is_shortened(full_url) else 0)
        self._add('HasBrandKeyword',      1 if self._has_brand_keyword(full_url) else 0)
        self._add('SuspiciousTLD',        1 if self._has_suspicious_tld(domain) else 0)
        self._add('PathEntropy',          round(self._entropy(path), 4))

        # ── 7. Additional engineered features ─────────────────────────

        # number of dots
        self._add('DotCount', full_url.count("."))

        # suspicious word count
        self._add('SuspiciousWordCount',
                sum(word in full_url.lower() for word in self.SUSPICIOUS_WORDS))

        # improved digit ratio
        digits = sum(c.isdigit() for c in full_url)
        self._add('DigitRatioImproved', digits / max(len(full_url),1))

        # port detection
        self._add('HasPort', 1 if ":" in parsed.netloc else 0)

        # max repeating character
        self._add('MaxCharRepeat', self._max_char_repeat(full_url))

        # entropy of domain
        self._add('DomainEntropy', round(self._entropy(domain),4))

        # path length
        self._add('PathLength', len(path))

        # query length
        self._add('QueryLength', len(query))

        # digit ratio in domain
        domain_digits = sum(c.isdigit() for c in domain)
        self._add('DomainDigitRatio', domain_digits / max(len(domain),1))

        # TLD risk score
        self._add('TLDRiskScore', self._tld_risk_score(domain))
        return self.features

    # ── helpers ──────────────────────────────────────────────────────────────

    def _add(self, name, value):
        self.features[name] = value

    def _is_ip(self, domain):
        return bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain))

    def _get_tld(self, domain):
        parts = domain.split('.')
        return '.' + parts[-1] if parts else ''

    def _count_subdomains(self, domain):
        parts = [p for p in domain.split('.') if p]
        return max(len(parts) - 2, 0)

    def _has_obfuscation(self, url):
        return bool(re.search(r'%[0-9a-fA-F]{2}', url))

    def _count_obfuscated(self, url):
        return len(re.findall(r'%[0-9a-fA-F]{2}', url))

    def _obfuscation_ratio(self, url):
        n = self._count_obfuscated(url)
        return round(n * 3 / max(len(url), 1), 4)

    def _count_special(self, url):
        special = set('!#$%^*()[]{}|\\<>,~`\'"')
        return sum(c in special for c in url)

    def _special_char_ratio(self, url):
        special = set('!@#$%^&*()[]{}|\\<>,~`\'"?=')
        return round(sum(c in special for c in url) / max(len(url), 1), 4)

    def _tld_legit_prob(self, domain):
        legit_tlds = {'.com': 0.9, '.org': 0.8, '.net': 0.75, '.edu': 0.95,
                      '.gov': 0.98, '.co': 0.7, '.io': 0.65, '.uk': 0.8,
                      '.de': 0.8, '.fr': 0.8, '.in': 0.75}
        tld = self._get_tld(domain)
        return legit_tlds.get(tld, 0.3 if tld in self.SUSPICIOUS_TLDS else 0.5)

    def _url_char_prob(self, url):
        if not url:
            return 0
        counts = Counter(url.lower())
        total = len(url)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c)
        return round(min(entropy / 8, 1), 4)

    def _char_continuation_rate(self, url):
        if len(url) < 2:
            return 0
        runs = sum(1 for i in range(1, len(url)) if url[i] == url[i-1])
        return round(runs / (len(url) - 1), 4)

    def _url_similarity_index(self, url):
        brand_score = sum(1 for b in self.BRAND_KEYWORDS if b in url.lower())
        return round(min(brand_score / 5, 1), 4)

    def _domain_title_score(self, domain, url):
        kws = [p for p in domain.split('.')[:-1] if len(p) > 3]
        hits = sum(1 for k in kws if k in url.lower())
        return round(hits / max(len(kws), 1), 4)

    def _url_title_score(self, url):
        kws = re.findall(r'[a-z]{4,}', url.lower())
        brand_hits = sum(1 for k in kws if k in self.BRAND_KEYWORDS)
        return round(brand_hits / max(len(kws), 1), 4)

    def _is_shortened(self, url):
        return any(s in url.lower() for s in self.SHORTENING_SERVICES)

    def _has_brand_keyword(self, url):
        return any(b in url.lower() for b in self.BRAND_KEYWORDS)

    def _has_suspicious_tld(self, domain):
        tld = self._get_tld(domain)
        return tld.lower() in self.SUSPICIOUS_TLDS

    def _entropy(self, text):
        if not text:
            return 0
        counts = Counter(text)
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def _max_char_repeat(self, text):
        max_run = 1
        run = 1

        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1

        return max_run
    
    def _tld_risk_score(self, domain):
        risk = {
            '.tk':0.9, '.ml':0.9, '.ga':0.9, '.cf':0.9,
            '.gq':0.9, '.xyz':0.7, '.top':0.8, '.club':0.6,
            '.online':0.6, '.site':0.6, '.live':0.5,
            '.com':0.1, '.org':0.1, '.net':0.2
        }

        tld = self._get_tld(domain)
        return risk.get(tld,0.4)

def get_feature_vector(url: str, feature_names: list = None) -> list:
    """Return feature values as an ordered list matching PhiUSIIL column order."""
    extractor = URLFeatureExtractor()
    feats = extractor.extract(url)
    if feature_names:
        return [feats.get(f, 0) for f in feature_names]
    return list(feats.values())


def get_feature_dict(url: str) -> dict:
    """Return full feature dictionary."""
    return URLFeatureExtractor().extract(url)


if __name__ == '__main__':
    test_urls = [
        'https://www.google.com/search?q=test',
        'http://paypal-secure-login.tk/verify?user=abc@gmail.com',
        'https://amazon.co.uk/dp/B09XYZ',
        'http://192.168.1.1/admin/login.php',
        'https://bit.ly/3xK9mNp',
    ]
    extractor = URLFeatureExtractor()
    for url in test_urls:
        feats = extractor.extract(url)
        print(f'\n--- {url[:60]} ---')
        for k, v in list(feats.items())[:12]:
            print(f'  {k:35s}: {v}')
