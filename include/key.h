#ifndef KEY_H_INCLUDED
#define KEY_H_INCLUDED

#include "types.h"
#include "hash.h"
#include "misc.h"
#include "platform.h"

namespace mra {

    /// Extracts the n'th bit as 0 or 1
    inline static Translation get_bit(int bits, Dimension n) {return ((bits>>n) & 0x1ul);}

    /// Extracts the low bit as 0 or 1
    inline static Translation low_bit(Translation l) {return l & Translation(1);}

    template <Dimension NDIM>
    class Key {
    private:
        Level n;  // = 0; cannot default initialize if want to be POD
        std::array<Translation,NDIM> l; // = {}; ditto

        /// Refreshes the hash value.  Note that the default std::hash does not mix enough
        SCOPE HashValue rehash() const {
            HashValue hashvalue = n;
            //for (Dimension d=0; d<NDIM; d++) mulhash(hashvalue,l[d]);
            for (Dimension d=0; d<NDIM; d++) hashvalue = (hashvalue<<7) | l[d];
            return hashvalue;
        }

    public:
        static constexpr int num_children() { return (1ul<<NDIM); }

        /// Default constructor is deliberately default so that is POD
        SCOPE Key() = default;

        /// Copy constructor default is OK
        SCOPE Key(const Key<NDIM>& key) = default;

        /// Move constructor default is OK
        SCOPE Key(Key<NDIM>&& key) = default;

        /// Construct from level and translation
        SCOPE Key(Level n, const std::array<Translation,NDIM>& l) : n(n), l(l)
        { }

        /// Construct from level and translation=0
        SCOPE Key(Level n) : n(n), l({0})
        { }

        /// Assignment default is OK
        SCOPE Key& operator=(const Key<NDIM>& other) = default;

        /// Move assignment default is OK
        SCOPE Key& operator=(Key<NDIM>&& key) = default;

        /// Equality
        SCOPE bool operator==(const Key<NDIM>& other) const {
            if (rehash() != other.rehash()) {
                return false;
            }
            else {
                if (n != other.n) return false;
                for (Dimension d = 0; d < NDIM; ++d) if (l[d] != other.l[d]) return false;
            }
            return true;
        }

        /// Inequality
        SCOPE bool operator!=(const Key<NDIM>& other) const {return !(*this == other);}

        /// Hash to unsigned value
        SCOPE HashValue hash() const {return rehash();}

        /// Level (n = 0, 1, 2, ...)
        SCOPE Level level() const {return n;}

        /// Translation (each element 0, 1, ..., 2**level-1)
        SCOPE const std::array<Translation,NDIM>& translation() const {return l;}

        /// Parent key

        /// Default is the immediate parent (generation=1).  To get
        /// the grandparent use generation=2, and similarly for
        /// great-grandparents.
        ///
        /// !! If there is no such parent it quietly returns the
        /// closest match (which may be self if this is the top of the
        /// tree).
        SCOPE Key<NDIM> parent(Level generation = 1) const {
            generation = std::min(generation,n);
            std::array<Translation,NDIM> pl;
            for (Dimension i=0; i<NDIM; i++) pl[i] = (l[i] >> generation);
            return Key<NDIM>(n-generation,pl);
        }

        /// First child in lexical ordering of KeyChildren iteration
        SCOPE Key<NDIM> first_child() const {
            assert(n<MAX_LEVEL);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x;
            return Key<NDIM>(n+1, l);
        }

        /// Last child in lexical ordering of KeyChildren iteration
        SCOPE Key<NDIM> last_child() const {
            assert(n<MAX_LEVEL);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x + 1;
            return Key<NDIM>(n+1, l);
        }

        /// Used by iterator to increment child translation
        SCOPE void next_child(int& bits) {
            int oldbits = bits++;
            for (Dimension d = 0; d < NDIM; ++d) {
                l[d] += get_bit(bits, d) - get_bit(oldbits,d);
            }
        }

        /// Map translation to child index in parent which is formed from binary code (bits)
        SCOPE int childindex() const {
            int b = low_bit(l[NDIM-1]);
            for (Dimension d=NDIM-1; d>0; d--) b = (b<<1) | low_bit(l[d-1]);
            return b;
        }

        /// Return the Key of the child at position idx \in [0, 1<<NDIM)
        SCOPE Key<NDIM> child_at(int idx) const {
            assert(n<MAX_LEVEL);
            assert(idx<num_children());
            std::array<Translation,NDIM> l = this->l;
            for (Dimension d = 0; d < NDIM; ++d) l[d] = 2*l[d] + ((idx & (1<<d)) ? 1 : 0);
            return Key<NDIM>(n+1, l);
        }

        SCOPE Key<NDIM> child_left(Dimension axis) const {
            assert(n<MAX_LEVEL);
            assert(axis < NDIM);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x;
            return Key<NDIM>(n+1, l);
        }

        SCOPE Key<NDIM> child_right(Dimension axis) const {
            assert(n<MAX_LEVEL);
            assert(axis < NDIM);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x;
            l[axis]++;
            return Key<NDIM>(n+1, l);
        }

        SCOPE bool is_left_child(Dimension axis) const {
            assert(n<MAX_LEVEL);
            return (this->l[axis] % 2) == 0;
        }

        SCOPE bool is_right_child(Dimension axis) const {
            assert(n<MAX_LEVEL);
            return (this->l[axis] % 2) == 1;
        }

        SCOPE Key<NDIM> neighbor(const Key<NDIM>& disp) const {
            std::array<Translation,NDIM> l = this->l + disp.l;
            return Key<NDIM>(n, l);
        }

        SCOPE Key<NDIM> neighbor(Dimension axis, int disp) const {
            if ((is_right_boundary(axis) && disp > 0) ||
                (is_left_boundary(axis) && disp < 0)) return invalid();
            std::array<Translation,NDIM> l = this->l;
            l[axis] += disp;
            return Key<NDIM>(n, l);
        }


        SCOPE bool is_left_boundary(Dimension axis) const {
            return (l[axis] == 0);
        }

        SCOPE bool is_right_boundary(Dimension axis) const {
            return (l[axis] == (1ul<<n)-1);
        }

        SCOPE bool is_boundary(Dimension axis) const {
            return is_left_boundary(axis) || is_right_boundary(axis);
        }

        SCOPE constexpr static Key<NDIM> invalid() {
            return Key<NDIM>(-1);
        }

        SCOPE constexpr bool is_invalid() const {
            return n == -1;
        }

        SCOPE constexpr bool is_valid() const {
            return n != -1;
        }
    };
    template <> inline SCOPE Key<1> Key<1>::parent(Level generation) const {
        generation = std::min(generation,n);
        return Key<1>(n-generation,{l[0]>>generation});
    }

    template <> inline SCOPE Key<2> Key<2>::parent(Level generation) const {
        generation = std::min(generation,n);
        return Key<2>(n-generation,{l[0]>>generation,l[1]>>generation});
    }

    template <> inline SCOPE Key<3> Key<3>::parent(Level generation) const {
        generation = std::min(generation,n);
        return Key<3>(n-generation,{l[0]>>generation,l[1]>>generation,l[2]>>generation});
    }

    template <> inline SCOPE Key<1> Key<1>::first_child() const {
        assert(n<MAX_LEVEL);
        return Key<1>(n+1, {l[0]<<1});
    }

    template <> inline SCOPE Key<2> Key<2>::first_child() const {
        assert(n<MAX_LEVEL);
        return Key<2>(n+1, {l[0]<<1,l[1]<<1});
    }

    template <> inline SCOPE Key<3> Key<3>::first_child() const {
        assert(n<MAX_LEVEL);
        return Key<3>(n+1, {l[0]<<1,l[1]<<1,l[2]<<1});
    }

    template <> inline SCOPE Key<1> Key<1>::last_child() const {
        assert(n<MAX_LEVEL);
        return Key<1>(n+1, {(l[0]<<1)+1});
    }

    template <> inline SCOPE Key<2> Key<2>::last_child() const {
        assert(n<MAX_LEVEL);
        return Key<2>(n+1, {(l[0]<<1)+1,(l[1]<<1)+1});

    }

    template <> inline SCOPE Key<3> Key<3>::last_child() const {
        assert(n<MAX_LEVEL);
        return Key<3>(n+1, {(l[0]<<1)+1,(l[1]<<1)+1,(l[2]<<1)+1});
    }

    template <> inline SCOPE void Key<1>::next_child(int& bits) {
        bits++; l[0]++;
        rehash();
    }

    template <> inline SCOPE void Key<2>::next_child(int& bits) {
        int oldbits = bits++;
        l[0] +=  (bits&0x1)     -  (oldbits&0x1);
        l[1] += ((bits&0x2)>>1) - ((oldbits&0x2)>>1);
        rehash();
    }

    template <> inline SCOPE void Key<3>::next_child(int& bits) {
        int oldbits = bits++;
        l[0] +=  (bits&0x1)     -  (oldbits&0x1);
        l[1] += ((bits&0x2)>>1) - ((oldbits&0x2)>>1);
        l[2] += ((bits&0x4)>>2) - ((oldbits&0x4)>>2);
        rehash();
    }

    template <> inline SCOPE int Key<1>::childindex() const {
        return l[0]&0x1;
    }

    template <> inline SCOPE int Key<2>::childindex() const {
        return ((l[1]&0x1)<<1) | (l[0]&0x1);
    }

    template <> inline SCOPE int Key<3>::childindex() const {
        return ((l[2]&0x1)<<2)  | ((l[1]&0x1)<<1) | (l[0]&0x1);
    }

    /// Range object used to iterate over children of a key
    template <Dimension NDIM>
    class KeyChildren {
        struct iterator {
            Key<NDIM> value;
            int bits;
            SCOPE iterator (const Key<NDIM>& value, int bits) : value(value), bits(bits) {}
            SCOPE operator const Key<NDIM>&() const {return value;}
            SCOPE const Key<NDIM>& operator*() const {return value;}
            SCOPE iterator& operator++() {
                value.next_child(bits);
                return *this;
            }
            SCOPE bool operator!=(const iterator& other) {return bits != other.bits;}

            /// Provides the index of the child (0, 1, ..., Key<NDIM>::num_children-1) while iterating
            SCOPE int index() const {return bits;}
        };
        iterator start, finish;

    public:
        SCOPE KeyChildren(const Key<NDIM>& key) : start(key.first_child(),0ul), finish(key.last_child(),(1ul<<NDIM)) {}
        SCOPE iterator begin() const {return start;}
        SCOPE iterator end() const {return finish;}
    };

    /// Returns range object for iterating over children of a key
    template <Dimension NDIM>
    SCOPE KeyChildren<NDIM> children(const Key<NDIM>& key) {return KeyChildren<NDIM>(key);}

    template <Dimension NDIM>
    std::ostream& operator<<(std::ostream& s, const Key<NDIM>& key) {
        s << "Key<" << int(NDIM) << ">(" << int(key.level()) << "," << key.translation() << ")";
        return s;
    }
}

namespace std {
    /// Ensures key satifies std::hash protocol
    template <mra::Dimension NDIM>
    struct hash<mra::Key<NDIM>> {
        SCOPE int operator()(const mra::Key<NDIM>& s) const noexcept { return s.hash(); }
    };
}


#endif
