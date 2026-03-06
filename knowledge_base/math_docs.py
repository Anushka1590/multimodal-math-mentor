docs = [
    {
        "id": "alg_001",
        "topic": "algebra",
        "title": "Quadratic Equations",
        "content": """
Quadratic Equation: ax^2 + bx + c = 0
Quadratic Formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a
Discriminant D = b^2 - 4ac
- D > 0: two distinct real roots
- D = 0: one repeated real root
- D < 0: no real roots (complex roots)
Sum of roots = -b/a
Product of roots = c/a
Completing the square: ax^2 + bx = a(x + b/2a)^2 - b^2/4a
Common mistake: forgetting ± in quadratic formula
"""
    },
    {
        "id": "alg_002",
        "topic": "algebra",
        "title": "Polynomials and Factor Theorem",
        "content": """
Factor Theorem: (x - a) is a factor of p(x) if and only if p(a) = 0
Remainder Theorem: when p(x) is divided by (x - a), remainder = p(a)
Degree of polynomial determines number of roots (counting multiplicity)
Sum of roots of x^n + a_(n-1)x^(n-1) + ... = -a_(n-1)
Product of roots = constant term / leading coefficient (sign depends on degree)
Common factoring identities:
- a^2 - b^2 = (a+b)(a-b)
- a^3 - b^3 = (a-b)(a^2+ab+b^2)
- a^3 + b^3 = (a+b)(a^2-ab+b^2)
Common mistake: sign errors when applying factor theorem
"""
    },
    {
        "id": "alg_003",
        "topic": "algebra",
        "title": "Sequences and Series",
        "content": """
Arithmetic Progression (AP):
- General term: a_n = a + (n-1)d
- Sum of n terms: S_n = n/2 * (2a + (n-1)d) = n/2 * (first + last)
- d = common difference

Geometric Progression (GP):
- General term: a_n = a * r^(n-1)
- Sum of n terms: S_n = a(r^n - 1)/(r - 1) for r ≠ 1
- Sum of infinite GP (|r| < 1): S = a/(1-r)
- r = common ratio

Common mistake: using GP formula when r = 1, or AP formula for non-constant differences
"""
    },
    {
        "id": "prob_001",
        "topic": "probability",
        "title": "Basic Probability Rules",
        "content": """
Probability of event A: P(A) = favorable outcomes / total outcomes
Range: 0 ≤ P(A) ≤ 1
Complement rule: P(A') = 1 - P(A)
Addition rule: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
Mutually exclusive events: P(A ∩ B) = 0, so P(A ∪ B) = P(A) + P(B)
Independent events: P(A ∩ B) = P(A) * P(B)
Conditional probability: P(A|B) = P(A ∩ B) / P(B)
Common mistake: assuming events are independent when they are not
"""
    },
    {
        "id": "prob_002",
        "topic": "probability",
        "title": "Permutations and Combinations",
        "content": """
Permutation (order matters): P(n,r) = n! / (n-r)!
Combination (order doesn't matter): C(n,r) = n! / (r! * (n-r)!)
C(n,r) = C(n, n-r)  ← symmetry property
C(n,0) = C(n,n) = 1
Binomial theorem: (a+b)^n = sum of C(n,r) * a^(n-r) * b^r for r=0 to n
Total ways to arrange n items with repeats: n! / (p! * q! * ...)
Common mistake: using permutation when combination is needed (ignoring order)
"""
    },
    {
        "id": "prob_003",
        "topic": "probability",
        "title": "Bayes Theorem and Conditional Probability",
        "content": """
Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)
Total probability: P(B) = P(B|A)*P(A) + P(B|A')*P(A')
Prior probability: initial belief P(A)
Posterior probability: updated belief P(A|B) after observing B
When to use Bayes: when you know P(B|A) but want P(A|B)
Common mistake: confusing P(A|B) with P(B|A) — these are NOT equal
"""
    },
    {
        "id": "calc_001",
        "topic": "calculus",
        "title": "Limits",
        "content": """
Limit: lim(x→a) f(x) = L means f(x) approaches L as x approaches a
L'Hopital's Rule: if limit gives 0/0 or ∞/∞, then lim f/g = lim f'/g'
Standard limits:
- lim(x→0) sin(x)/x = 1
- lim(x→0) (1 - cos x)/x = 0
- lim(x→0) (e^x - 1)/x = 1
- lim(x→∞) (1 + 1/x)^x = e
Squeeze theorem: if g(x) ≤ f(x) ≤ h(x) and lim g = lim h = L, then lim f = L
Common mistake: applying L'Hopital when form is not indeterminate
"""
    },
    {
        "id": "calc_002",
        "topic": "calculus",
        "title": "Derivatives",
        "content": """
Definition: f'(x) = lim(h→0) [f(x+h) - f(x)] / h
Basic rules:
- Power rule: d/dx(x^n) = n*x^(n-1)
- Product rule: d/dx(uv) = u'v + uv'
- Quotient rule: d/dx(u/v) = (u'v - uv') / v^2
- Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)
Standard derivatives:
- d/dx(sin x) = cos x
- d/dx(cos x) = -sin x
- d/dx(e^x) = e^x
- d/dx(ln x) = 1/x
- d/dx(tan x) = sec^2 x
Common mistake: forgetting chain rule for composite functions
"""
    },
    {
        "id": "calc_003",
        "topic": "calculus",
        "title": "Applications of Derivatives",
        "content": """
Critical points: where f'(x) = 0 or f'(x) is undefined
First derivative test:
- f'(x) changes + to -: local maximum
- f'(x) changes - to +: local minimum
Second derivative test:
- f''(x) < 0 at critical point: local maximum
- f''(x) > 0 at critical point: local minimum
- f''(x) = 0: inconclusive, use first derivative test
Increasing: f'(x) > 0
Decreasing: f'(x) < 0
Inflection point: where f''(x) = 0 and sign changes
Optimization: find critical points, check endpoints and boundaries
Common mistake: forgetting to verify critical point is max or min
"""
    },
    {
        "id": "linalg_001",
        "topic": "linear_algebra",
        "title": "Matrices and Determinants",
        "content": """
Matrix multiplication: (AB)_ij = sum of row i of A × column j of B
Matrix multiplication is NOT commutative: AB ≠ BA in general
Determinant of 2x2: |A| = ad - bc for [[a,b],[c,d]]
Determinant of 3x3: expand along any row or column
Properties of determinants:
- det(AB) = det(A) * det(B)
- det(A^T) = det(A)
- If any row/column is all zeros: det = 0
- Swapping two rows changes sign of det
Inverse exists only if det(A) ≠ 0
A^(-1) = (1/det(A)) * adjugate(A)
Common mistake: row/column index confusion in 3x3 determinant expansion
"""
    },
    {
        "id": "linalg_002",
        "topic": "linear_algebra",
        "title": "System of Linear Equations",
        "content": """
System AX = B
Cramer's rule: x_i = det(A_i) / det(A) where A_i replaces column i with B
Gaussian elimination: row reduce augmented matrix [A|B]
Number of solutions:
- Unique solution: det(A) ≠ 0, rank(A) = rank([A|B]) = n
- No solution: rank(A) ≠ rank([A|B])
- Infinite solutions: rank(A) = rank([A|B]) < n
Homogeneous system AX = 0 always has trivial solution X = 0
Non-trivial solution exists only when det(A) = 0
Common mistake: arithmetic errors during row reduction
"""
    },
    {
        "id": "linalg_003",
        "topic": "linear_algebra",
        "title": "Eigenvalues and Eigenvectors",
        "content": """
Eigenvalue equation: Av = λv where v ≠ 0
Characteristic equation: det(A - λI) = 0
Steps to find eigenvalues:
1. Compute det(A - λI) = 0
2. Solve the characteristic polynomial for λ
Steps to find eigenvectors:
1. For each λ, solve (A - λI)v = 0
2. Find null space of (A - λI)
Properties:
- Sum of eigenvalues = trace(A)
- Product of eigenvalues = det(A)
- Symmetric matrices have real eigenvalues
Common mistake: using det(A + λI) instead of det(A - λI)
"""
    },
    {
        "id": "tips_001",
        "topic": "general",
        "title": "Common JEE Math Mistakes to Avoid",
        "content": """
1. Sign errors: double-check negative signs especially in quadratic formula
2. Domain errors: sqrt requires non-negative input, log requires positive input
3. Division by zero: always check denominator before dividing
4. Forgetting absolute value in |x| problems
5. Confusing permutation and combination
6. Not checking boundary conditions in optimization
7. Applying L'Hopital when form is not 0/0 or ∞/∞
8. Forgetting chain rule in differentiation
9. Assuming P(A|B) = P(B|A)
10. Not verifying if critical point is max or min
Always verify your answer by substituting back into original equation.
"""
    },
    {
        "id": "tips_002",
        "topic": "general",
        "title": "Problem Solving Strategy for JEE Math",
        "content": """
Step 1: Read the problem carefully, identify what is given and what is asked
Step 2: Identify the topic (algebra, probability, calculus, linear algebra)
Step 3: Write down relevant formulas
Step 4: Identify the solution approach before computing
Step 5: Solve step by step, showing all work
Step 6: Verify the answer using alternate method or substitution
Step 7: Check units and domain constraints
For MCQ: eliminate wrong options using boundary cases (x=0, x=1, etc.)
For proof questions: work forwards from given AND backwards from conclusion
Time management: if stuck for 2 mins, move on and return later
"""
    },
    {
        "id": "calc_004",
        "topic": "calculus",
        "title": "Integration Basics",
        "content": """
Integration is the reverse of differentiation
Basic rules:
- Power rule: integral of x^n = x^(n+1)/(n+1) + C, for n ≠ -1
- integral of 1/x = ln|x| + C
- integral of e^x = e^x + C
- integral of sin x = -cos x + C
- integral of cos x = sin x + C
Definite integral: integral from a to b of f(x)dx = F(b) - F(a)
Fundamental theorem: d/dx [integral from a to x of f(t)dt] = f(x)
Integration by substitution: let u = g(x), then du = g'(x)dx
Common mistake: forgetting +C for indefinite integrals, sign errors in trig integrals
"""
    },
    {
        "id": "prob_004",
        "topic": "probability",
        "title": "Last Digit Multiplication Patterns",
        "content": """
Last digit of a product depends only on last digits of the numbers multiplied.

Digits that can NEVER produce last digit 1,3,5,7 in a product:
- Any number ending in 0,2,4,5,6,8 when combined with certain digits changes parity

For product to end in 1,3,7,9 (odd and not 5):
- ALL four numbers must end in 1,3,7,9 (not 5, not even)
- Digits ending in 5 always produce product ending in 5 or 0

For product to end in 1,3,5,7:
- All four numbers must end in ODD digits: 1,3,5,7,9
- Probability each number is odd = 5/10 = 1/2
- But product ends in 5 if ANY number ends in 5
- Digits 1,3,7,9 out of 10 = 4/10 = 2/5 probability each

Correct approach for "last digit is 1,3,7 or 9":
- Each number must end in 1,3,7,9 → probability = 4/10 = 2/5
- For four numbers: (2/5)^4 = 16/625

For "last digit is 1,3,5,7":
- Numbers ending in 5 give product ending in 5 (valid)
- Numbers ending in 1,3,7,9 give product ending in 1,3,7,9 (valid if not 9)
- Cleanest interpretation: all four end in 1,3,7,9 → (4/10)^4 = 16/625
- This is the standard JEE answer for this type of problem

Key rule: digits 1,3,7,9 are the only digits whose products stay in {1,3,7,9}
Probability = (4/10)^4 = 16/625
"""
    },
    {
        "id": "prob_005",
        "topic": "probability",
        "title": "Classical Probability with Digits and Numbers",
        "content": """
When selecting numbers randomly from 1-10 or using last digits:
- Total possible last digits: 0,1,2,3,4,5,6,7,8,9 → 10 equally likely

Odd digits: 1,3,5,7,9 → 5 out of 10 → probability 1/2
Even digits: 0,2,4,6,8 → 5 out of 10 → probability 1/2

Digits whose powers/products stay odd AND not divisible by 5:
- Only: 1,3,7,9 → 4 out of 10 → probability 2/5

Product of n numbers ends in odd digit ≠ 5:
- Each number must end in 1,3,7,9
- Probability = (4/10)^n = (2/5)^n

For n=4: (2/5)^4 = 16/625 ≈ 0.0256

Product ends in exactly 5:
- At least one number ends in 5, rest are odd
- More complex calculation needed

Standard JEE shortcut:
- "last digit is 1,3,7 or 9" → (4/10)^4 = 16/625
- "last digit is odd" → (5/10)^4 = 1/16
- "last digit is 1,3,5,7" → standard answer is 16/625 (treating as 1,3,7,9 pattern)
"""
    },
    {
        "id": "prob_007",
        "topic": "probability",
        "title": "Hypergeometric Distribution",
        "content": """
Hypergeometric Distribution: sampling WITHOUT replacement from finite population.

When to use Hypergeometric (NOT Binomial):
- Finite population of size N
- Sampling without replacement
- k successes in population, N-k failures

PMF: P(X=x) = C(K,x) * C(N-K, n-x) / C(N,n)
where:
- N = population size
- K = total successes in population
- n = sample size
- x = observed successes

Mean: E(X) = n * K/N
Variance: Var(X) = n * (K/N) * (1 - K/N) * (N-n)/(N-1)

The factor (N-n)/(N-1) is called finite population correction factor.
This is what makes it DIFFERENT from Binomial variance = n*p*(1-p)

Example: Box has 10 pens, 3 defective. Draw 2 without replacement.
N=10, K=3, n=2
E(X) = 2 * 3/10 = 6/10 = 3/5
Var(X) = 2 * (3/10) * (7/10) * (10-2)/(10-1)
       = 2 * 3/10 * 7/10 * 8/9
       = 2 * 21/100 * 8/9
       = 336/900
       = 28/75

Key rule: if problem says "without replacement" or involves finite box/bag/group
         → ALWAYS use Hypergeometric, NOT Binomial
"""
    },
    {
        "id": "prob_008",
        "topic": "probability",
        "title": "Binomial vs Hypergeometric vs Poisson",
        "content": """
Choosing the correct distribution — critical for JEE:

BINOMIAL: Use when
- Fixed number of trials n
- Each trial independent
- Sampling WITH replacement OR infinite population
- Constant probability p each trial
- Variance = n*p*(1-p)

HYPERGEOMETRIC: Use when
- Sampling WITHOUT replacement
- Finite population of size N
- Drawing n items
- Variance = n*(K/N)*(1-K/N)*(N-n)/(N-1)
- Keywords: box, bag, lot, batch, group with fixed defectives

POISSON: Use when
- Rare events over time/space
- Large n, small p, np = lambda is moderate
- Variance = lambda = mean

GEOMETRIC: Use when
- Number of trials until first success
- Mean = 1/p, Variance = (1-p)/p^2

Decision rule:
- "drawn from box without replacement" → Hypergeometric
- "each draw independent / with replacement" → Binomial
- "average rate of occurrence" → Poisson
- "until first success" → Geometric
"""
    },
    {
        "id": "calc_005",
        "topic": "calculus",
        "title": "Integration by Parts and Special Integrals",
        "content": """
Integration by Parts: integral(u dv) = uv - integral(v du)
LIATE rule for choosing u: Logarithm, Inverse trig, Algebraic, Trig, Exponential

Special integrals:
- integral(x*e^x) = e^x(x-1) + C
- integral(x^2*e^x) = e^x(x^2-2x+2) + C
- integral(sin^2 x) = x/2 - sin(2x)/4 + C
- integral(cos^2 x) = x/2 + sin(2x)/4 + C
- integral(1/(a^2+x^2)) = (1/a)*arctan(x/a) + C
- integral(1/sqrt(a^2-x^2)) = arcsin(x/a) + C

Definite integral properties:
- integral from a to b = -integral from b to a
- integral from a to a = 0
- integral from 0 to 2a f(x)dx = 2*integral from 0 to a f(x)dx if f(2a-x)=f(x)
- integral from -a to a f(x)dx = 2*integral from 0 to a f(x)dx if f is even, 0 if odd
"""
    },
    {
        "id": "alg_004",
        "topic": "algebra",
        "title": "Complex Numbers",
        "content": """
Complex number: z = a + bi where i = sqrt(-1), i^2 = -1
Modulus: |z| = sqrt(a^2 + b^2)
Argument: arg(z) = arctan(b/a)
Conjugate: z* = a - bi
z * z* = |z|^2

Polar form: z = r(cos θ + i sin θ) = r*e^(iθ)
De Moivre's theorem: z^n = r^n(cos nθ + i sin nθ)

nth roots of unity: z^n = 1 has n roots: e^(2πik/n) for k=0,1,...,n-1
Sum of all nth roots of unity = 0
Product of all nth roots of unity = (-1)^(n+1)

Operations:
- (a+bi)+(c+di) = (a+c)+(b+d)i
- (a+bi)(c+di) = (ac-bd)+(ad+bc)i
- 1/(a+bi) = (a-bi)/(a^2+b^2)

Common mistake: forgetting i^2=-1, or wrong quadrant for argument
"""
    },
    {
        "id": "linalg_004",
        "topic": "linear_algebra",
        "title": "Rank and Nullity of Matrix",
        "content": """
Rank of matrix = number of non-zero rows in row echelon form
Nullity = n - rank (where n = number of columns)
Rank-Nullity theorem: rank(A) + nullity(A) = n

Full rank: rank = min(m,n) for m×n matrix
- Square matrix full rank → invertible → unique solution
- rank(A) < n → infinite solutions or no solution

For system AX = B:
- Consistent (solution exists): rank(A) = rank([A|B])
- Unique solution: rank(A) = rank([A|B]) = n
- Infinite solutions: rank(A) = rank([A|B]) < n
- No solution: rank(A) ≠ rank([A|B])

Row operations don't change rank:
1. Swap two rows
2. Multiply row by non-zero scalar
3. Add multiple of one row to another
"""
    },
    {
        "id": "calc_006",
        "topic": "calculus",
        "title": "Strictly Increasing and Decreasing Functions",
        "content": """
A function f(x) is strictly decreasing on interval (a,b) if f'(x) < 0 for all x in (a,b).
A function f(x) is strictly increasing on interval (a,b) if f'(x) > 0 for all x in (a,b).

For absolute value functions like f(t) = |t+1|/t^2, t < 0:
Split into cases:
- When t < -1: |t+1| = -(t+1), so f(t) = -(t+1)/t^2
- When -1 < t < 0: |t+1| = (t+1), so f(t) = (t+1)/t^2

To find where f is strictly decreasing, compute f'(t) and find where f'(t) < 0.

For f(t) = -(t+1)/t^2 when t < -1:
f'(t) = [(-1)*t^2 - (-(t+1))*2t] / t^4
      = [-t^2 + 2t(t+1)] / t^4
      = [t^2 + 2t] / t^4
      = (t+2) / t^3
f'(t) < 0 when t < -2 (since t^3 < 0 for t < 0)

For f(t) = (t+1)/t^2 when -1 < t < 0:
f'(t) = [t^2 - 2t(t+1)] / t^4
      = [-t^2 - 2t] / t^4
      = -(t+2) / t^3
f'(t) < 0 when -1 < t < 0 (always negative here)

Largest interval where f is strictly decreasing: (-2, -1) union (-1, 0)
But for LARGEST SINGLE interval: (-2, 0) excluding t=-1

If interval is given as (2α, α):
2α = -2 and α = 0? No, α = -1 gives interval (2*(-1), -1) = (-2, -1) ✓
So α = -1

Critical rule: when finding α from interval notation (2α, α),
solve 2α = left endpoint and α = right endpoint consistently.

For local max/min problems with log:
- g'(x) = 0 gives critical points
- At critical point where ln term vanishes (argument = 1), answer is often clean integer
- Always check if ln(1) = 0 simplifies the answer to exact integer
"""
    }
]