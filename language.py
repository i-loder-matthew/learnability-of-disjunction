import itertools



import numpy as np

# God knows what else I'll need (also feel like this is probably more disgusting than it should be -- will work on trying to simplify models further)



class Language():


    # irrealis: 0, 1 = irr agreement, phrasal irr, both
    # int_disj: 0, 1, 2, 3 = none, conjunction, form 1, form 2
    # std_disj: 0, 1, 2, 3 = none, conjunction, form 1, form 2
    # int_markers: 0, 1, 2 = none, int, irr
    # std_markers: 0, 1, 2 = none, int, irr
    # props: inventory of propositional letters

    def __init__(self, irrealis, int_disj, std_disj, int_markers, std_markers, props = ["a","b"]) -> None:
        
        # self._p_irr = "maybe" if irrealis != 0 else "" # yeah.. I'm gonna get rid of this one. 
        self._m_irr = "%" if irrealis == 1 else ""
        self._int_disj = "" if int_disj == 0 else "&" if int_disj == 1 else "*" if int_disj == 2 else "^"
        self._std_disj = "" if std_disj == 0 else "&" if std_disj == 1 else "*" if std_disj == 2 else "^"
        self._int_marker = "" if int_markers == 0 else "?" if int_markers == 1 else self._m_irr
        self._std_marker = "" if std_markers == 0 else "?" if std_markers == 1 else self._m_irr
        self._props = props
        self._simple = False

        if len(self._props) == 2:
            self._simple = True


    def generate_str_from_model(self, model):

        # model is a numpy array

        if self._simple:
            p1 = self._props[0]
            p2 = self._props[1]

            info_request = model[0,0] == 1
            n_both = sum(model[:,1])
            n_p1 = n_both + sum(model[:,2])
            n_p2 = n_both + sum(model[:,3])
            n_none = 1 - n_both

            EOS = ["<eos>"]
            SOS = ["<sos>"]

            # Still missing case where only one is possible...
            if n_both > 0.8 * len(model):
                return SOS + [p1, "&", p2] + EOS if not info_request else SOS + EOS
            elif n_p1 > 0.8 * len(model):
                if n_p2 > 0.2 * len(model):
                    if info_request:
                        return SOS + [p2, "?"] + EOS
                    else:
                        return SOS + [p1, "&", p2, self._m_irr] + EOS
                else:
                    return SOS + [p1] + EOS if not info_request else SOS + [p2, "?"] + EOS
            elif n_p2 > 0.8 * len(model):
                if n_p1 > 0.2 * len(model):
                    if info_request:
                        return SOS + [p1, "?"] + EOS
                    else:
                        return SOS + [p2, "&", p1, self._m_irr] + EOS
                else:
                    return SOS + [p2] + EOS if not (info_request) else SOS + [p1, "?"] + EOS
            else:
                if info_request:
                    return SOS + [p1, self._int_marker, self._int_disj, p2, self._int_marker] + EOS
                else:
                    if n_none >= 0.8 * len(model):
                        return SOS + EOS
                    elif n_both > 0.2 * len(model):
                        return SOS + [p1, self._std_marker, self._std_disj, p2, self._std_marker] + EOS
                    elif n_p1 > 0.2 * len(model):
                        return SOS + [p1, self._m_irr] + EOS
                    elif n_p2 > 0.2 * len(model):
                        return SOS + [p2, self._m_irr] + EOS
                    else:
                        return SOS + EOS         
        

    




