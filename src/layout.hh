#ifndef LAYOUT_HH
#define LAYOUT_HH

#include <tuple>
#include <memory>
#include <ostream>

namespace sys {

	template <class T>
	struct wrap {
		typedef T type;
	};

	template<class ... Args>
	struct Sentence;

	template<>
	struct Sentence<> {
		template <class Instance>
		void
		print(const Instance&, std::ostream&) const {}

		template <class Instance>
		void
		read(Instance&, std::ostream&) {}
	};

	template<class T, class ... Args>
	struct Sentence<T, Args...>: public Sentence<Args...> {

		typedef typename std::tuple_element<2,T>::type field_type;
		typedef const typename field_type::type* pointer;

		explicit
		Sentence(const T& k, const Args& ... args):
		Sentence<Args...>(args...),
		_word(k)
		{}

		Sentence(const Sentence&) = default;
		Sentence(Sentence&&) = default;

		template <class Instance>
		void
		print(const Instance& instance, std::ostream& out) const {
			const char* base = reinterpret_cast<const char*>(std::addressof(instance));
			const char* name = std::get<0>(_word);
			const size_t offset = std::get<1>(_word);
			out << name
				<< '='
				<< *reinterpret_cast<pointer>(base + offset)
				<< '\n';
			static_cast<const Sentence<Args...>*>(this)->print(instance, out);
		}

		template <class Instance>
		void
		read(Instance& instance, std::ostream& in) {
			const char* base = reinterpret_cast<const char*>(std::addressof(instance));
			const char* name = std::get<0>(_word);
			const size_t offset = std::get<1>(_word);
			in >> *reinterpret_cast<pointer>(base + offset);
			static_cast<Sentence<Args...>*>(this)->read(instance, in);
		}

	private:
		T _word;
	};

	template <class T, class S>
	class Instance {
		const T& _instance;
		const S& _sentence;

	public:
		Instance(const T& inst, const S& sn):
		_instance(inst),
		_sentence(sn)
		{}

		friend std::ostream&
		operator<<(std::ostream& out, const Instance& rhs) {
			rhs._sentence.print(rhs._instance, out);
			return out;
		}
	};

	template<class ... Args>
	Sentence<Args...>
	make_sentence(const Args& ... args) {
		return Sentence<Args...>(args...);
	}

	template<class ... Args>
	std::shared_ptr<Sentence<Args...>>
	make_sentence_ptr(const Args& ... args) {
		return std::shared_ptr<Sentence<Args...>>(new Sentence<Args...>(args...));
	}

	template<class T, class ... Args>
	Instance<T, Sentence<Args...>>
	make_instance(const T& instance, const Args& ... args) {
		return Instance<T, Sentence<Args...>>(instance, make_sentence(args...));
	}

}

#endif // LAYOUT_HH
