so maybe i want to compute a Range for each bucket

let's define a Range as a tuple of probabilities R = (p1, p2, ..., pn) where
pa = probability of taking action a in A (A is the set of defined action buckets)

since this Range is used for evaluation, i can make it pretty complicated in terms of
action choices. so, for the set of actions A i can take it as this to start (x = bet size):
A = {
    Fold
    Check
    Call
    Bet(0 < x <= 3bb)
    Bet(3bb < x <= 9bb)
    Bet(9bb < x <= 27bb)
    Bet(27bb < x <= 54bb)
    Bet(54bb < x <= 108bb)
    Bet(108bb < x)
    Bet(0 < x <= 1/4pot)
    Bet(1/4pot < x <= 1/3pot)
    Bet(1/3pot < x <= 1/2pot)
    Bet(1/2pot < x <= 2/3pot)
    Bet(3/4pot < x <= pot)
    Bet(pot < x <= 3/2pot)
    Bet(3/2pot < x)
}
potential additions: x as a fraction of stack depth

now, for each game state (under some state abstraction), i will compute these Ranges
and use them for evaluation.

ok, so what state abstraction?

state features bucket on: {
    position
    street
    effective SPR
    number of players in hand
    Board Texture Abstraction
    Preflop Action Abstraction
},

where:
"Board Texture Abstraction" is identifying similar boards together
(e.g. 'wet board', 'dry board', 'monotone board', etc)
"Preflop Action Abstraction" is identifying similar preflop action histories
(e.g. '3bet pot', 'limped pot', 'single-raise pot', etc)



Ok, this looks great!!!!!

I want to get into the practical side now

Maybe it makes sense to design classes in Python to represent all this structure

I think it's reasonable to start with the action abstraction part, since the Range is pretty simple once the actions are known.

So how should I structure it? I want to make it seamless to hot-swap different abstractions, since i will surely have to fiddle with different levels
of granularity, remove/add features, etc
