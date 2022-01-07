extern "C"
{
  #include "Modules/text.h"
  #include "Modules/sync.h"
}

#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace
{

  void text_from_string(Text* text, std::wstring a)
  {
    for (auto c: a)
      append_char(text, /* suspect = */false, c);
  }

  std::wstring text_to_string(Text* text)
  {
    std::wstring w;
    w.reserve(text->count);
    for (auto* item = text->first; item; item = item->next)
      w.push_back(item->value);
    return w;
  }

  int compute_mapping(Synclist* synclist, int* A, int* B)
  {

    int i = 0;
    for (Sync* sync = synclist->first; sync; sync = sync->next)
    {
      int ma = sync->substr[0].length;
      int mb = sync->substr[1].length;
      int m = std::max(ma, mb);
      for (int k = 0; k < ma; ++k)
        A[sync->substr[0].start + k] = i + k;
      for (int k = 0; k < mb; ++k)
        B[sync->substr[1].start + k] = i + k;
      i += m;
    }
    return i;
  }

}


int compute_align_map(std::wstring a, std::wstring b, std::vector<int>& map_A, std::vector<int>& map_B)
{
  Text A;
  Text B;
  list_initialize(&A);
  list_initialize(&B);
  text_from_string(&A, a);
  text_from_string(&B, b);

  map_A.resize(a.size());
  map_B.resize(b.size());

  int alignment_size;
  {
    Text texts[2] = {A, B};

    Synclist synclist;
    fastukk_sync(&synclist, texts);
    alignment_size = compute_mapping(&synclist, map_A.data(), map_B.data());
  }

  list_empty(&A, free);
  list_empty(&B, free);

  return alignment_size;
}

std::pair<std::vector<int>, std::vector<int>> align_2(std::wstring a, std::wstring b)
{
  std::vector<int> map_A, map_B;
  int alignment_size = compute_align_map(a, b, map_A, map_B);

  return std::make_pair(map_A, map_B);
}


std::pair<std::wstring, std::wstring> align(std::wstring a, std::wstring b, wchar_t delchr)
{
  std::vector<int> map_A, map_B;
  int alignment_size = compute_align_map(a, b, map_A, map_B);


  std::wstring A(alignment_size, delchr);
  std::wstring B(alignment_size, delchr);

  for (std::size_t i = 0; i < map_A.size(); ++i)
    A[map_A[i]] = a[i];

  for (std::size_t i = 0; i < map_B.size(); ++i)
    B[map_B[i]] = b[i];

  return std::make_pair(A, B);
}




PYBIND11_MODULE(isri_tools, m) {
    m.doc() = "ISRI Analytic tools"; // optional module docstring
    m.def("align", &align, "Align two strings");
    m.def("get_align_map", &align_2, "Get the mapping between two strings");
}
