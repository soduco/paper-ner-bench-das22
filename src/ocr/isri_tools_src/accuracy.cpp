
/**********************************************************************
 *
 *  accuracy.c
 *
 *  Author: Stephen V. Rice
 *
 * Copyright 1996 The Board of Regents of the Nevada System of Higher
 * Education, on behalf, of the University of Nevada, Las Vegas,
 * Information Science Research Institute
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License.  You
 * may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 *
 **********************************************************************/

extern "C"
{
#include "Modules/accrpt.h"
#include "Modules/sync.h"
}

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <string>
#include <fstream>

namespace py = pybind11;

namespace
{
constexpr int MAX_DISPLAY = 24;

void
make_key (Text *texts, char *key, Sync *sync)
{
  long i, j;
  char buffer[2][MAX_DISPLAY + 4], string[STRING_SIZE];
  for (i = 0; i < 2; i++)
    {
      buffer[i][0] = '\0';
      for (j = sync->substr[i].start; j <= sync->substr[i].stop; j++)
        {
          char_to_string (False, texts[i].array[j]->value, string, True);
          if (strlen (buffer[i]) + strlen (string) > MAX_DISPLAY)
            {
              strcat (buffer[i], "...");
              break;
            }
          strcat (buffer[i], string);
        }
    }
  sprintf (key, "{%s}-{%s}\n", buffer[0], buffer[1]);
}
/**********************************************************************/

void
add_ops (Accops *sum_ops, Accops *ops)
{
  sum_ops->ins += ops->ins;
  sum_ops->subst += ops->subst;
  sum_ops->del += ops->del;
  sum_ops->errors += ops->errors;
}

void
process_synclist (Text *texts, Synclist *synclist, Accdata *accdata)
{
  Sync *sync;
  long i, characters, wildcards, reject_characters, suspect_markers, genchars;
  Accops ops;
  char key[100];
  for (sync = synclist->first; sync; sync = sync->next)
    {
      characters = wildcards = 0;
      for (i = sync->substr[0].start; i <= sync->substr[0].stop; i++)
        if (texts[0].array[i]->value == REJECT_CHARACTER)
          wildcards++;
        else
          {
            characters++;
            add_class (accdata, texts[0].array[i]->value, 1, (sync->match ? 0 : 1));
          }
      accdata->characters += characters;
      reject_characters = suspect_markers = 0;
      for (i = sync->substr[1].start; i <= sync->substr[1].stop; i++)
        if (texts[1].array[i]->value == REJECT_CHARACTER)
          reject_characters++;
        else if (texts[1].array[i]->suspect)
          suspect_markers++;
      accdata->reject_characters += reject_characters;
      accdata->suspect_markers += suspect_markers;
      if (sync->match)
        accdata->false_marks += suspect_markers;
      else
        {
          genchars = std::max (0L, sync->substr[1].length - wildcards);
          ops.errors = std::max (characters, genchars);
          if (ops.errors > 0)
            {
              accdata->errors += ops.errors;
              ops.ins = std::max (0L, characters - genchars);
              ops.subst = std::min (characters, genchars);
              ops.del = std::max (0L, genchars - characters);
              make_key (texts, key, sync);
              if (reject_characters + suspect_markers > 0)
                {
                  add_ops (&(accdata->marked_ops), &ops);
                  add_conf (accdata, key, ops.errors, ops.errors);
                }
              else
                {
                  add_ops (&(accdata->unmarked_ops), &ops);
                  add_conf (accdata, key, ops.errors, 0);
                }
              add_ops (&(accdata->total_ops), &ops);
            }
        }
    }
}

void
text_from_string (Text *text, std::wstring a)
{
  for (auto c : a)
    append_char (text, /* suspect = */ false, c);
}

} // namespace

std::unique_ptr<Accdata>
get_accuracy_stats (std::wstring ref, std::wstring prediction)
{
  Text A;
  Text B;
  list_initialize (&A);
  list_initialize (&B);
  text_from_string (&A, ref);        // 16 bit fixed encoding only for now
  text_from_string (&B, prediction); // 16 bit fixed encoding only for now

  Text texts[2] = { A, B };
  Synclist synclist;
  fastukk_sync (&synclist, texts); // Actual call to original UNLV/ISRI tools

  auto data = std::make_unique<Accdata> ();
  process_synclist (texts, &synclist, data.get());

  return data;
}

std::wstring
print_accurary_report (Accdata *data)
{
  FILE* tmp = tmpfile();
  int old_desc = dup (fileno (stdout));
  dup2 (fileno(tmp), fileno (stdout));
  write_accrpt (data, nullptr);
  dup2 (old_desc, fileno (stdout));

  rewind(tmp);
  std::wstring result;
  wchar_t c;
  while ((c = fgetwc(tmp)) != EOF)
    result.push_back(c);

  fclose(tmp);
  return result;
}

void
init_accuracy (py::module &m)
{
  py::class_<Accdata> (m, "AccStat")
    .def_readonly ("characters", &Accdata::characters, "number of ground-truth characters")
    .def_readonly ("errors", &Accdata::errors, "number of errors made")
    .def_readonly ("reject_characters", &Accdata::reject_characters, "number of reject characters generated")
    .def_readonly ("suspect_markers", &Accdata::suspect_markers, "number of characters marked as suspect")
    .def_readonly ("false_marks", &Accdata::false_marks, "number of false marks")
    .def ("__repr__", &print_accurary_report);
  m.def ("compute_accurary_stats", &get_accuracy_stats);
}
