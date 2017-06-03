#include "layout.hh"
#include <string>
#include <cstddef>
#include <iostream>

class My_class {
	int field1 = 123;
	double field2 = 1.23;
	std::string field3 = "hello";

public:
	static decltype(auto)
	layout() {
		return sys::make_sentence(
			std::make_tuple("field1", offsetof(My_class, field1), sys::wrap<int>()),
			std::make_tuple("field2", offsetof(My_class, field2), sys::wrap<decltype(field2)>()),
			std::make_tuple("field3", offsetof(My_class, field3), sys::wrap<decltype(field3)>())
		);
	}

	void
	print() {
		layout().print(*this, std::cout);
	}
};

int main() {
	My_class obj;
	obj.print();
	return 0;
}
