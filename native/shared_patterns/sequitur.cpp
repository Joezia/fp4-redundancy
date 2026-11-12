// Adapted 2026-11-09 from craignm/sequitur b6afb6e6049cc7e54284f8f346604937b3c09296.
// Apache-2.0, full LICENSE in third_party/references/sequitur.
// Core check/substitute/expand and K=1 are retained. Buffer input, grammar output,
// scoped allocation cleanup and an STL digram index replace CLI/global storage.
#include "grammar.hpp"
#include <fstream>
#include <cassert>
#include <iostream>
namespace sequitur_native {
class symbols;class rules;
std::unordered_set<symbols*> live_symbols;
std::unordered_set<rules*> live_rules;
std::map<std::pair<unsigned long,unsigned long>,symbols*> digrams;
int num_rules=0,num_symbols=0,current_rule=0,K=1;
int delimiter=-1,do_uncompress=0,numbers=0,print_rule_usage=0,max_rule_len=2;
bool compression_initialized=true;
/****************************************************************************

 classes.h - Contains definitions of 'rule' and 'symbols' classes, and the
             bodies of some of their methods.

****************************************************************************/


using namespace std;

extern int num_symbols, current_rule, K;
extern int occupied, table_size;

class symbols;
class rules;
ostream &operator << (ostream &o, symbols &s);

typedef unsigned long ulong;

extern symbols **find_digram(symbols *s);     // defined in classes.cc

///////////////////////////////////////////////////////////////////////////

class rules {

  // the guard node in the linked list of symbols that make up the rule
  // It points forward to the first symbol in the rule, and backwards
  // to the last symbol in the rule. Its own value points to the rule data
  // structure, so that symbols can find out which rule they're in
  symbols *guard;

  // count keeps track of the number of times the rule is used in the grammar
  int count;

  // Usage stores the number of times a rule is used in the input.
  //    An example of the difference between count and Usage: in a grammar
  //    with rules S->abXcdXef , X->gAiAj , A->kl , rule A's count is 2
  //    (it is used two times in the grammar), while its Usage is 4 (there
  //    are two X's in the input sequence, and each of them uses A two times)
  int Usage;

  // number can serve two purposes:
  // (1) numbering the rules nicely for printing
  //     (in this case it's not essential for the algorithm)
  // (2) if this is a non-terminal symbol, assign a code to it
  //     (used in compression and forget_print())
  int number;

public:
  void output();     // output right hand of the rule when printing out grammar
  void output2();    // output right hand of the rule when compressing

  rules();
  ~rules();

  void reuse() { count ++; }
  void deuse() { count --; }

  symbols *first();     // pointer to first symbol of rule's right hand
  symbols *last();      // pointer to last symbol of rule's right hand

  int freq()           { return count; }
  int usage()          { return Usage; }
  void usage(int i)    { Usage += i; }
  int index()          { return number; }
  void index(int i)    { number = i; }

  void reproduce();    // reproduce full expansion of the rule
};

class symbols {
  symbols *n, *p;     // next and previous symbol within the rule
  ulong s;            // symbol value (e.g. ASCII code, or rule index)

public:

  // print out symbol, or, if it is non-terminal, rule's full expansion
  void reproduce() {
    extern int numbers;

    if (non_terminal()) rule()->reproduce();
    else {
      cout << *this;
      if (numbers) cout << ' ';
    }
  }

  // initializes a new terminal symbol
  symbols(ulong sym) {
    s = sym * 2 + 1; // an odd number, so that they're a distinct
                     // space from the rule pointers, which are 4-byte aligned
    p = n = 0;
    num_symbols ++; live_symbols.insert(this);
  }

  // initializes a new symbol to refer to a rule, and increments the reference
  // count of the corresponding rule
  symbols(rules *r) {
    s = (ulong) r;
    p = n = 0;
    rule()->reuse();
    num_symbols ++; live_symbols.insert(this);
  }

  // links two symbols together, removing any old digram from the hash table
  static void join(symbols *left, symbols *right) {
    if (left->n) {
      left->delete_digram();

      // This is to deal with triples, where we only record the second
      // pair of the overlapping digrams. When we delete the second pair,
      // we insert the first pair into the hash table so that we don't
      // forget about it.  e.g. abbbabcbb

      if (right->p && right->n &&
          right->value() == right->p->value() &&
          right->value() == right->n->value()) {
        assert(find_digram(right));
        symbols **r = find_digram(right);
        if (r) { // necessary when using delimiters
          *r = right;
          occupied ++;
        }
      }

      if (left->p && left->n &&
          left->value() == left->n->value() &&
          left->value() == left->p->value()) {
        symbols **lp = find_digram(left->p);
        if (lp) { // necessary when using delimiters
          *lp = left->p;
          occupied ++;
        }
      }
    }
    left->n = right; right->p = left;
  }

  // cleans up for symbol deletion: removes hash table entry and decrements
  // rule reference count
  ~symbols() {
    live_symbols.erase(this);
    join(p, n);
    if (!is_guard()) {
      delete_digram();
      if (non_terminal()) rule()->deuse();
    }
    num_symbols --;
  }

  // inserts a symbol after this one.
  void insert_after(symbols *y) {
    join(y, n);
    join(this, y);
  }

  // removes the digram from the hash table
  void delete_digram() {
    if (is_guard() || n->is_guard()) return;
    symbols **m = find_digram(this);
    if (m == 0) return;
    for (int i = 0; i < K; i ++)
      if (m[i] == this) {
	m[i] = (symbols *) 1;
	occupied --;
      }
  }

  // is_guard() returns true if this is the guard node
  // marking the beginning/end of a rule
  bool is_guard() { return non_terminal() && rule()->first()->prev() == this; }

  // non_terminal() returns true if a symbol is non-terminal.
  // We make sure that terminals have odd-numbered values.
  // Because the value of a non-terminal is a pointer to
  // the corresponding rule, they're guaranteed to be even
  // (more -- a multiple of 4) on modern architectures

  int non_terminal() { return ((s % 2) == 0) && (s != 0);}

  symbols *next() { return n; }
  symbols *prev() { return p; }
  inline ulong raw_value() { return s; }
  inline ulong value() { return s / 2; }

  // assuming this is a non-terminal, rule() returns the corresponding rule
  rules *rule() { return (rules *) s; }

  // substitute digram with non-terminal symbol
  void substitute(rules *r);

  // check digram and enforce the Sequitur constraints if necessary
  int check();

  // substitute non-terminal symbol with its rule's right hand
  void expand();

  void point_to_self() { join(this, this); }

};
/****************************************************************************

 classes.cc - Module containing (part of the) methods of 'rules' and 'symbols'
              classes, and functions for working with the hash table of digrams
              printing out the grammar.

 Notes:
    For the rest of 'symbols' and 'rules' methods, see classes.h .

 ****************************************************************************/


extern int num_rules, delimiter, do_uncompress;

rules::rules() {
  num_rules ++; live_rules.insert(this);
  guard = new symbols(this);
  guard->point_to_self();
  count = number = Usage = 0;
}

rules::~rules() {
  live_rules.erase(this);
  num_rules --;
  delete guard;
}

// pointer to first symbol of rule's right hand
symbols *rules::first() { return guard->next(); }
// pointer to  last symbol of rule's right hand
symbols *rules::last()  { return guard->prev(); }

// **************************************************************************
// symbols::check()
//    check digram made of this symbol and the symbol following it,
//    and enforce Sequitur constraints.
//
// Return values
//    0 : did not change the grammar (there was no violation of contraints)
//    1 : did change the grammar (there were violations of contraints)
//
// Global variables used
//    K (minimum number of times a digram must occur to form rule)
// **************************************************************************
int symbols::check() {
  if (is_guard() || n->is_guard()) return 0;

  symbols **x = find_digram(this);
  // if either symbol of the digram is a delimiter -> do nothing
  if (!x) return 0;

  int i;

  // if digram is not yet in the hash table -> put it there, and return
  for (i = 0; i < K; i ++)
    if (ulong(x[i]) <= 1) {
      x[i] = this;
      occupied ++;
      return 0;
    }

  // if repetitions overlap -> do nothing
  for (i = 0; i < K; i ++)
    if (x[i]->next() == this || next() == x[i])
      return 0;

  rules *r;

  // reuse an existing rule

  for (i = 0; i < K; i ++)
    if (x[i]->prev()->is_guard() && x[i]->next()->next()->is_guard()) {
      r = x[i]->prev()->rule();
      substitute(r);

      // check for an underused rule

      if (r->first()->non_terminal() && r->first()->rule()->freq() == 1)
	r->first()->expand();
      // if (r->last()->non_terminal() && r->last()->rule()->freq() == 1)
      //   r->last()->expand();

      return 1;
    }

  symbols *y[100];
  for (i = 0; i < K; i ++) y[i] = x[i];

  // make a copy of the pointers to digrams,
  // so that they don't change under our feet
  // especially when we create this replacement rule

  // create a new rule

  r = new rules;

  if (non_terminal())
    r->last()->insert_after(new symbols(rule()));
  else
    r->last()->insert_after(new symbols(value()));

  if (next()->non_terminal())
    r->last()->insert_after(new symbols(next()->rule()));
  else
    r->last()->insert_after(new symbols(next()->value()));

  for (i = 0; i < K; i ++) {
    if (y[i] == r->first()) continue;
    // check that this hasn't been deleted
    bool deleted = 1;
    for (int j = 0; j < K; j ++)
      if (y[i] == x[j]) {
        deleted = 0;
        break;
      }
    if (deleted) continue;

    y[i]->substitute(r);
    //    y[i] = (symbols *) 1; // should be x
  }

  x[0] = r->first();
  occupied ++;

  substitute(r);

  // check for an underused rule

  if (r->first()->non_terminal() && r->first()->rule()->freq() == 1)
    r->first()->expand();
  //  if (r->last()->non_terminal() && r->last()->rule()->freq() == 1)
  //    r->last()->expand();

  return 1;
}

// ***************************************************************************
// symbols::expand()
//    This symbol is the last reference to its rule. It is deleted, and the
//    contents of the rule substituted in its place.
// ***************************************************************************
void symbols::expand() {
  symbols *left = prev();
  symbols *right = next();
  symbols *f = rule()->first();
  symbols *l = rule()->last();

  extern bool compression_initialized;
  if (!compression_initialized) {
    int i = 0;
    symbols *s;
    extern int max_rule_len;
    // first calculate length of this rule (the one we are adding symbols to)
    // symbol 'this' should not be counted because it will be deleted
    s = next();
    do {
       if (!s->is_guard()) i++;
       s = s->next();
    } while(s != this);
    // then calculate length of what is to be added
    for (s = f; !s->is_guard(); s = s->next()) i++;
    if (i > max_rule_len) max_rule_len = i;
  }

  symbols **m = find_digram(this);
  if (!m) return;
  delete rule();

  for (int i = 0; i < K; i ++)
    if (m[i] == this) {
      m[i] = (symbols *) 1;
      occupied --;
    }

  s = 0; // if we don't do this, deleting the symbol tries to deuse the rule!

  delete this;

  join(left, f);
  join(l, right);

  symbols **ll = find_digram(l);
  if (ll) {
    *ll = l;
    occupied ++;
  }
}

// ***************************************************************************
// symbols::substitute(rules *r)
//    Replace digram made up of this symbol and the symbol following it with
//    a non-terminal, which points to rule "r" (parameter).
// ***************************************************************************
void symbols::substitute(rules *r)
{
  symbols *q = p;

  delete q->next();
  delete q->next();

  q->insert_after(new symbols(r));

  if (!q->check()) q->next()->check();
}

int table_size;
int lookups = 0;
int collisions = 0;
int occupied = 0;
symbols **table = 0;

// ***************************************************************************
// symbols **find_digram(symbols *s)
//
//     Search hash table for digram made of symbols s->value() and
//     s->next()->value().
//
// Return value
//   - if digram found : Pointer to hash table element where digram is stored.
//   - otherwise       : 0
//
// Global variables used
//    delimiter  (symbol accross which not to form rules, see sequitur.cc)
// ***************************************************************************
symbols **find_digram(symbols *s) {
  auto key=std::make_pair(s->raw_value(),s->next()->raw_value());
  return &digrams[key];
}

// **************************************************************************
// rules::reproduce()
//    Reproduce full expansion of a rule.
// **************************************************************************
void rules::reproduce()
{
  // for each symbol of the rule, call symbols::reproduce()!
  for (symbols *p = first(); !p->is_guard(); p = p->next())
    p->reproduce();
}


// **************************************************************************
// Overload operator << to write symbols of the grammar to streams,
//    in a formatted manner.
// **************************************************************************
ostream &operator << (ostream &o, symbols &s)
{
  extern int numbers;

  if (s.non_terminal())
     o << s.rule()->index();
  else if (numbers & do_uncompress) o << s.value() << endl;
  else if (numbers) o << '[' << s.value() << ']';
  else if (do_uncompress) o << char(s.value());
  else if (s.value() == '\n') o << "\\n";
  else if (s.value() == '\t') o << "\\t";
  else if (s.value() == ' ' ) o << '_';
  else if (s.value() == '\\' ||
       s.value() == '(' ||
       s.value() == ')' ||
       s.value() == '_' ||
       isdigit(s.value()))
    o << '\\' << char(s.value());
  else o << char(s.value());

  return o;
}

// **************************************************************************
// rules::output()

//    Print right hand of this rule. If right hand contains non-terminals,
//    descend recursively and print right hands of subordinated rules, if
//    these have not been printed out yet.
//
//    Also, numbers the rule, which will indicate that right hand has been
//    printed out.
//
// Global variables used
//    current_rule, print_rule_usage
// **************************************************************************
void rules::output()
{
  symbols *s;

  for (s = first(); !s->is_guard(); s = s->next())
     if (s->non_terminal() && s->rule()->index() == 0)
        s->rule()->output();

  number = current_rule ++;

  for (s = first(); !s->is_guard(); s = s->next())
    cout << *s << ' ';

  extern int print_rule_usage;
  if (print_rule_usage) cout << "\t(" << Usage << ")";

  cout << endl;
}

}
namespace patterns {
Stream sequitur_encode(const Words& words,unsigned alphabet) {
    static std::mutex mutex;std::lock_guard<std::mutex> lock(mutex);
    using namespace sequitur_native;
    struct Cleanup {~Cleanup(){
        // Plain pointer/int objects: release every remaining allocation without
        // running list-mutating destructors against a partially destroyed grammar.
        for(auto p:live_symbols)::operator delete(p);
        for(auto p:live_rules)::operator delete(p);
        live_symbols.clear();live_rules.clear();digrams.clear();num_rules=num_symbols=occupied=0;
    }} cleanup;
    auto start=new rules;
    for(auto v:words){start->last()->insert_after(new symbols(static_cast<unsigned long>(v)));start->last()->prev()->check();}
    std::vector<rules*> list{start};std::unordered_map<rules*,uint32_t> ids{{start,0}};
    for(size_t i=0;i<list.size();++i)for(auto p=list[i]->first();!p->is_guard();p=p->next())if(p->non_terminal()&&!ids.count(p->rule())){
        ids[p->rule()]=list.size();list.push_back(p->rule());}
    Grammar g;g.rules.resize(list.size()-1);
    for(size_t i=0;i<list.size();++i){auto& rhs=i?g.rules[i-1]:g.root;
        for(auto p=list[i]->first();!p->is_guard();p=p->next()){
            if(p->non_terminal()){auto id=ids.at(p->rule());if(!id)throw std::runtime_error("Sequitur references start rule");rhs.push_back(alphabet+id-1);}
            else rhs.push_back(p->value());}}
    return grammar_stream(g,words.size());
}
}
