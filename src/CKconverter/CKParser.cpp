/**
 *  @file CKParser.cpp
 *
 */

// Copyright 2001  California Institute of Technology
//
// $Log: CKParser.cpp,v $
// Revision 1.20  2005/12/09 17:49:34  dggoodwin
// removed critical and saturation properties from ThermoPhase
//
// Revision 1.19  2005/07/28 23:02:44  hkmoffa
// Got rid of one warning message.
//
// Revision 1.18  2005/07/26 03:56:35  dggoodwin
// cleanup
//
// Revision 1.17  2005/07/25 03:51:21  dggoodwin
// now recognizes the FORD keyword
//
// Revision 1.16  2005/01/07 10:26:43  dggoodwin
// merged changes from branch
//
// Revision 1.15.2.2  2004/12/18 15:16:13  dggoodwin
// minor cleanup
//
// Revision 1.15.2.1  2004/12/18 15:00:02  dggoodwin
// *** empty log message ***
//
//
// Revision 1.6  2004/07/02 16:48:13  hkmoffa
// Moved CK_SyntaxError definition to the .h file. It's used in more
// than one .cpp file.
//
// Revision 1.5  2004/05/13 17:45:05  dggoodwin
// fixed bug in which a species name beginning with M was interpreted as a third body
//
// Revision 1.4  2004/05/13 16:58:33  dggoodwin
// *** empty log message ***
//
//

// turn off warnings about truncating long names under Windows
#ifdef _WIN32
#pragma warning(disable:4786)
#endif

#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "CKParser.h"
#include "ckr_utils.h"
#include "writelog.h"
#include "atomicWeightDB.h" 
#include <stdio.h>
//#include "../stringUtils.h"

using namespace std;

namespace ckr {


    static string int2s(int n, string fmt="%d") {
        char buf[30];
        sprintf(buf, fmt.c_str(), n);
        return string(buf);
    }

    /// Exception class for syntax errors.
     CK_SyntaxError::CK_SyntaxError(ostream& f, 
				    const string& s, int linenum) 
	 : m_out(f) {
	 m_msg += "Syntax error: " + s;
	 if (linenum > 0) m_msg += "  (line " + int2s(linenum) + ")\n";
     }


    static int parseGroupString(string str, vector<string>& esyms, 
        vector_int& result);

    /**
     *  Throw an exception if one of the four lines that must have 
     *  1, 2, 3, or 4 in column 80 do not.
     */
    static void illegalThermoLine(ostream& f, 
        char n, int linenum = -1) 
    {
        throw CK_SyntaxError(f, "column 80 must "
            "contain an integer", linenum); 
    };


    /**
     *  Throw an exception if one of the four lines is less than 
     *  80 columns
     */
    static void illegalThermoLineTooShort(ostream& f, 
        int linenum = -1) 
    {
        throw CK_SyntaxError(f, "thermo data must "
            "be at least 80 columns in length", linenum); 
    };


    /**
     *  Throw an exception if number string is bad
     */
    static void illegalNumber(ostream& f, 
        string s, int linenum = -1) 
    {
        string msg = "illegal number: "+s;
        throw CK_SyntaxError(f, msg, linenum); 
    };


  // MJM mod 2011-03-05: add atomicWeightDB.cpp as a header file for 
  // CKParser.h
  //extern void getDefaultAtomicWeights(map<string,double>& weights);
  void getDefaultAtomicWeights(map<string, double>& weights) {
        
        // erase existing entries, if any
        //weights.clear();
        const int MAX_NUM = 200;
        int n;
        for (n = 0; n < MAX_NUM; n++) {
            if (_symbols[n][0] == '!') break;
            weights[_symbols[n]] = _weights[n];
        }
    }
    static string d2e(string s) {
        size_t n;
        size_t sz = s.size();
        string r = s;
        char ch;
        for (n = 0; n < sz; n++) {
            ch = s[n];
            if (ch == 'D') r[n] = 'E';
            else if (ch == 'd') r[n] = 'e';
        }
        return r;
    }

    static double de_atof(string s) {
        string r = d2e(s);
	//double rval = Cantera::atofCheck(r.c_str()); 
        double rval = atof(r.c_str());
        return rval;
    }

    static double getNumberFromString(string s) {
        bool inexp = false;
        removeWhiteSpace(s);
        int sz = static_cast<int>(s.size());
        char ch;
        for (int n = 0; n < sz; n++) {
            ch = s[n];
            if (!inexp && (ch == 'E' || ch == 'e' || ch == 'D' || ch == 'd')) 
                inexp = true;
            else if (ch == '+' || ch == '-') {
                if (n > 0 && (s[n-1] != 'E' && s[n-1] 
                        != 'e' && s[n-1] != 'd' && s[n-1] != 'D')) {
                    return UNDEF;
                }
            }
            else if (ch != '.' && (ch < '0' || ch > '9')) {
                return UNDEF;
            }
        }
        return de_atof(s);
    }

    /**
     *  Add an element to a species.
     *  @param symbol  element symbol
     *  @param atoms   number of atoms of this element in the 
     *                 species (may be non-integral)
     *  @param sp      Species object to add element to
     *  @param log     log file output stream
     */
    static void addElement(string symbol, double atoms, 
        Species& sp, ostream& log) {
        
        if (atoms != 0.0) {
            Constituent e;
            e.name = symbol;
            e.number = atoms;
            sp.elements.push_back(e);
            sp.comp[symbol] = atoms;
        }
    }


    /**
     *  Check validity of the three temperatures defining the two 
     *  temperature ranges for the NASA polynomial species thermodynamic
     *  property fits. 
     *  @param log     log file output stream
     *  @param tmin    minimum temperature
     *  @param tmid    intermediate temperature
     *  @param tmax    maximum temperature
     */
    static void checkTemps(ostream& log, double tmin, 
        double tmid, double tmax) 
    {
        if (tmin == 0.0 || tmid == 0.0 || tmax == 0.0) {
            throw CK_SyntaxError(log, 
                "error reading Tmin, Tmid, or Tmax");
        }
    }

    static void getSpecies(string s, 
        int n, vector<RxnSpecies>& species, bool debug, ostream& log) {
        removeWhiteSpace(s);
        // break string into substrings at the '+' characters separating
        // species symbols
        int i, nn;
        bool inplus = true;
        vector<int> pluses;
        vector<string> sp;
        for (i = n-1; i >= 0; i--) {
            if (!inplus && s[i] == '+') {
                pluses.push_back(i);
                inplus = true;
            }
            else if (inplus && s[i] != '+') {
                inplus = false;
            }
        }
        pluses.push_back(-1);
        int np = pluses.size();
        int loc, nxt;
        for (nn = 0; nn < np; nn++) {
            loc = pluses.back();
            pluses.pop_back();
            if (nn == np-1) nxt = s.size();
            else nxt = pluses.back();
            sp.push_back(s.substr(loc+1,nxt-loc-1));
        }

        int ns = sp.size();
        string r, num;
        int sz, j, strt=0;
        RxnSpecies ss;
        for (nn = 0; nn < ns; nn++) {
            r = sp[nn];
            sz = r.size();
            for (j = 0; j < sz; j++) {
                if (!((r[j] >= '0' && r[j] <= '9') || r[j] == '.')) {
                    strt = j;
                    break;
                }
            }
            ss.name = r.substr(strt,sz); 
            if (strt == 0)
                ss.number = 1.0;
            else
                ss.number = atof(r.substr(0,strt).c_str());
            species.push_back(ss);
            if (debug) {
                log << ss.number << "  " << ss.name << endl;
            }
        }
    }


    /**
     *  given a string specifying either the reactant or product side of a
     *  reaction equation, construct a list of Constituent objects
     *  containing the species symbols and stoichiometric coefficients.
     *  @todo allow non-integral stoichiometric coefficients
     */
    int getGroups(string::const_iterator begin, string::const_iterator end, vector<string>& esyms, 
        vector<grouplist_t>& rxngroups) 
    {
        bool ingroup = false;
        rxngroups.clear();
        string g = "";
        group_t igrp;
        grouplist_t groups;

        for (; begin != end; ++begin) {
            if (*begin == '(') {
                ingroup = true;
                g = "";
                }
            else if (*begin == ')') {
                ingroup = false;
                igrp.clear();
                if (parseGroupString(g, esyms, igrp) >= 0)
                    groups.push_back(igrp);
                else return -1;
            }
            else if (*begin == '+') {
                rxngroups.push_back(groups);
                groups.clear();
            }
            else if (ingroup && *begin != ' ') {
                g += *begin;
            }
        }
        rxngroups.push_back(groups);
        return 1;
    }
  static bool isFinalReactionValid(Reaction &rxn,
                                   ostream* parser_log,
                                   int line_num) {
    switch(rxn.type) {

    case PLogInterpolation:
      // correct the line number to report the reaction line
      line_num-=rxn.kf.pres_pts_plog.size()+1;
      if(rxn.kf.pres_pts_plog.size() < 1) {
        throw CK_SyntaxError(*parser_log,
	  "Error: at least one pressure point is needed for PLOG",
	   line_num);
        return false; 
      }
      if(rxn.isFalloffRxn) {
        throw CK_SyntaxError(*parser_log,
	  "Error: PLOG and falloff reactions (LOW/TROE/SRI) are mutually exclusive",
	   line_num);
        return false; 
      }
      if(rxn.isChemActRxn) {
        throw CK_SyntaxError(*parser_log,
	   "Error: PLOG and chemical activation reactions (HIGH/TROE/SRI) are mutually exclusive",
	   line_num);
        return false; 
      }
      if(rxn.isThreeBodyRxn) {
        throw CK_SyntaxError(*parser_log,
	   "Error: PLOG and 3rd-body reactions M or (M) are mutually exclusive",
	   line_num);
        return false; 
      }
      if(rxn.isReversible && rxn.krev.A > 0.0) {
        throw CK_SyntaxError(*parser_log,
	   "Error: PLOG does not support REV parameters being specified",
	   line_num);
        return false; 
      }
      
      return true;
    default:
      if(rxn.kf.type != Arrhenius) {
        throw CK_SyntaxError(*parser_log,
	   "Error: Only Arrhenius and PLogInterpolation supported for forward rate coefficients",
	   line_num);
        return false; 
      }
      if(rxn.krev.type != Arrhenius) {
        throw CK_SyntaxError(*parser_log,
	   "Error: Only Arrhenius supported for reverse rate coefficients",
	   line_num);
        return false; 
      }
      if(rxn.isFalloffRxn && rxn.isThreeBodyRxn) {
        throw CK_SyntaxError(*parser_log,
	   "Error: Falloff and 3rd-body reactions are treated as exclusive classes",
	   line_num);
        return false; 
      }

      if(rxn.isThreeBodyRxn) {
        if(rxn.thirdBody.compare("M")!=0) {
          throw CK_SyntaxError(*parser_log,
            "Error: reactions has a third body != M",
	   line_num);
          return false; 
        }
      }
      if(rxn.isFalloffRxn) {
        if(rxn.falloffType == Troe) {
          int nParams = rxn.falloffParameters.size();
          if(nParams != 3 && nParams != 4) {
            throw CK_SyntaxError(*parser_log,
              "Error: Troe reaction must have 3 or 4 parameters",
             line_num);
            return false; 
          }
        } else if( rxn.falloffType != SRI && rxn.falloffType != Lindemann ) {
          throw CK_SyntaxError(*parser_log,
            "Error: Unsupported falloff reaction type",
           line_num);
          return false; 
        }
      }
      return true;
    }
  }
    /**
     * Constructor. Construct a parser for the specified input file.
     */
    CKParser::CKParser(istream* infile, const string& fname, ostream* log) 
        : verbose(true), debug(false), m_line (0) {
        m_ckfile = infile; 
        m_ckfilename = fname;
        m_log = log;
        m_nasafmt = false;
        m_last_eol = '\n';
    }


    /**
     * 
     *  Get a line from the input file, and return it in string s. If the
     *  line contains a comment character (!), then return only the
     *  portion preceding it.  Non-printing characters are replaced by
     *  spaces.  
     *  @param s On return, s contains the line read from the
     *  input file.
     *  @param comment On return, comment contains the text following the
     *  comment character on the line, if any.
     *
     */
    void CKParser::getCKLine(string& s, string& comment) {

        // Chemkin comment character    
        const char commentChar = '!';

        // Cantera anti-comment character
        const char undoCommentChar = '%';

        // carriage return
        const char char13 = char(13);

        // linefeed
        const char char10 = char(10);

        istream& f = *m_ckfile;

        // if putCKLine was called to 'put back' a line, then return this
        // line, instead of reading a new one

        if (!m_buf.empty()) {
            s = m_buf;
            m_buf = "";
            comment = m_comment;
            m_comment = "";
            return;
        }

        // read a line, convert non-printing characters to ' ', 
        // and remove comments

        comment = "";
        string line;

        line = "";
        char ch = ' ';
        while (1 > 0) {
            f.get(ch);
            if (!f || f.eof()) break;

            // convert tabs to spaces
            if (ch == '\t') ch = ' ';

            // if an end-of-line character is seen, then break.
            // Check for all common end-of-line characters.
            if (ch == char13 || (ch == char10 && (m_last_eol != char13)))  {
#undef DEBUG_EOL
#ifdef DEBUG_EOL
                *m_log << "EOL: found character " << int(ch) << " ending line:" << endl;
                *m_log << line << endl;
                *m_log << int(m_last_eol) << " " << int('\n') << " " << int(ch) << endl; 
#endif
                m_last_eol = ch;
                break;
            }
#if _WIN32
            if (ch>=0 && ch < 255 && isprint(ch)) line += ch;
#else
            if (isprint(ch)) line += ch;
#endif
        }

        string::size_type icom = line.find(commentChar);

        // lines that begin with !% are not comments for Cantera
        if (icom == 0 && line[1] == undoCommentChar) {
            line[0] = '%';
            line[1] = ' ';
            icom = line.find(commentChar);
        }
        int len = static_cast<int>(line.size());

        for (int i = 0; i < len; i++) if (!isprint(line[i])) line[i] = ' ';
        if (icom != string::npos) {
            s = line.substr(0, icom);
            comment = line.substr(icom+1,len-icom-1); 
        }
        else {
            s = line;
        }

        if (!f || f.eof()) {
            s = "<EOF>";
            comment = "";
            return;
        }
        m_line++;
    }


    /**
     *
     *   Put back a line read from the input file. The next call to
     *   getCKLine will return this line.
     *
     */

    void CKParser::putCKLine(string& s, string& comment) {
        m_buf = s;
        m_comment = comment;
    }


    bool CKParser::advanceToKeyword(const string& kw, const string& stop) {
        string s, c;
        do {
            getCKLine(s,c);
            if (match(s,"<EOF>")) return false;
            if (match(s,kw)) {
                putCKLine(s,c);
                return true;
            }
        }
        while (!match(s,stop));
        putCKLine(s,c);
        return false;
    }



    /**
     *
     *  Read the element section of the input file, and return
     *  a list of the elements found.
     *
     */
    
    bool CKParser::readElementSection(elementList& elements) {
        string s, comment;
        int firsttok;
        vector<string> toks;

        map<string,double> defaultWeights; 
        //ct::ctmap_sd defaultWeights; 
        getDefaultAtomicWeights(defaultWeights);

        //istream& f = m_ckfile;
        int ntok;

        elements.clear();
        while (1 > 0) {

next:
            if (advanceToKeyword("ELEM", "SPEC")) {
                firsttok = 1;
                while (1 > 0) {
                    do {
                        getCKLine(s, comment);
                        getTokens(s, static_cast<int>(s.size()), toks);
                        ntok = static_cast<int>(toks.size());
                    } while (ntok == 0);

                    if (firsttok == 0 && isKeyword(toks[0])) {
                        putCKLine(s,comment);
                        goto next;
                    }
                    for (int i = firsttok; i < ntok; i++) {
                        if (match(toks[i],"END")) goto next;
                        else {                            
                            Element el;
                            string wtstr;
                            el.comment = comment;
                            el.index = static_cast<int>(elements.size());
                            if (extractSlashData(toks[i], el.name, wtstr)) {
                                el.atomicWeight = de_atof(wtstr);
                                el.weightFromDB = false;
                            }
                            else {
                                el.atomicWeight = defaultWeights[capitalize(el.name)];
                                el.weightFromDB = true;
                            }
                            if (el.atomicWeight > 0.0) el.valid = 1;
                            else el.valid = 0;
                            if (find(elements.begin(), 
                                elements.end(), el) < elements.end()) {
                                if (m_log) 
                                    *m_log << "warning... duplicate element " 
                                           << el.name << " (ignored)." << endl;
                            }
                            else 
                                elements.push_back(el);
                        }
                    }
                    firsttok = 0;
                }
            }
            else {
                if (elements.size() == 0) {
                    *m_log << "no elements found." << endl;
                    return false;
                }
                else return valid(elements);
            }
        }
        return false;
    }



    /**
     *
     *  Read the SPECIES section of the input file, and return
     *  a list of species names. 
     * 
     */
    bool CKParser::readSpeciesSection(speciesList& species) {
        string s, comment;
        int firsttok;
        vector<string> toks;

        int ntok;
        int nsp = 0;

        while (1 > 0) {

 next:
            if (advanceToKeyword("SPEC", "THER")) {
                firsttok = 1;
                while (1 > 0) {
                    do {
                        getCKLine(s, comment);
                        getTokens(s, static_cast<int>(s.size()), toks);
                        ntok = static_cast<int>(toks.size());
                    } while (ntok == 0);

                    if (firsttok == 0 && isKeyword(toks[0])) {
                        putCKLine(s,comment);
                        goto next;
                    }
                    for (int i = firsttok; i < ntok; i++) {
                        if (match(toks[i],"END")) goto next;
                        else {
                            Species sp;
                            sp.name = toks[i];
                            if (find(species.begin(), species.end(), sp) 
                                < species.end()) {
                                if (m_log)
                                    *m_log << "warning... duplicate species " 
                                           << sp.name << " (ignored)." << endl;
                            }
                            else {
                                nsp++;
                                sp.index = nsp;
                                species.push_back(sp);
                            }
                        }
                    }
                    firsttok = 0;
                }
            }
            else {
                if (species.size() == 0) return false;
                else return true;
            }
        }
        return false;
    }



    /**
     *
     * Read species data from THERMO section records. 
     * 
     * @param names List of species names (input).
     * @param species  Table of species objects holding data from records
     * in THERMO section (output). 
     * @param allowExtThermoData True if 'THERMO' specified, false if 
     * 'THERMO ALL' specified.
     *
     */

    bool CKParser::readThermoSection(vector<string>& names, 
        speciesTable& species, vector_fp& temp, 
        int& optionFlag, ostream& log) {
        string s;
        vector<string> toks;

        double tmin = -1.0, tmid = -1.0, tmax = -1.0;
        if (temp.size() == 3) {
            tmin = temp[0];
            tmid = temp[1];
            tmax = temp[2];
        }

        int nsp = static_cast<int>(names.size());

        string comment;

        // read lines until THERMO section is found. But if EOF reached or 
        // start of REACTIONS section, then there is no THERMO section.
        do {
            getCKLine(s,comment);
            if (match(s,"<EOF>")) return false;
            if (match(s,"REAC")) {
                putCKLine(s,comment);
                return false;
            }
        }
        while (!match(s,"THER"));

        // read the tokens on the THERMO line
        getTokens(s, static_cast<int>(s.size()), toks);
        m_nasafmt = false;
        if (toks.size() >= 2)
        { 
            unsigned int itt;
            for (itt = 1; itt < toks.size(); itt++) {
                if (match(toks[itt],"ALL")) {
                    optionFlag = NoThermoDatabase;
                }
                else if (match(toks[itt],"NO_TMID")) {
                    m_nasafmt = true;
                    log << "\nOption 'NO_TMID' specified. Default midpoint temperature\n";
                    log << "will be used for all species.\n\n";
                }
                else throw CK_SyntaxError(log, 
                    "unrecognized THERMO option.", m_line);
            }
        }

        // if "THERMO ALL" specified, or if optionFlag is set to HasTempRange,
        // then the next line must be the 3 default temperatures for the database.

        if (optionFlag == NoThermoDatabase || optionFlag == HasTempRange) {
            getCKLine(s, comment);
            getTokens(s, static_cast<int>(s.size()), toks);
            if (toks.size() >= 3) {
                tmin = de_atof(toks[0]);
                tmid = de_atof(toks[1]);
                tmax = de_atof(toks[2]);
            }
        
            if (verbose) {
                log.flags(ios::showpoint | ios::fixed);
                log.precision(2);
                log << endl << " default Tlow, Tmid, Thigh: " << tmin << "  " 
                    << tmid << "  " << tmax << endl;
            }
            checkTemps(log, tmin, tmid, tmax);
            temp.clear();
            temp.push_back(tmin);
            temp.push_back(tmid);
            temp.push_back(tmax);
        }

        // now read in all species records that have names in list 'names'

        bool getAllSpecies = (nsp > 0 && match(names[0],"<ALL>"));
        if (getAllSpecies) names.clear();

        map<string, int> dup; // used to check for duplicate THERMO records
        bool already_read;

        while (1 > 0) {
            if (nsp == 0) break;
            already_read = false;

            Species spec;
            Species storedSpec;
            readThermoRecord(spec);

            if (spec.name == "<END>") {
                break;
            } 
            // check for duplicate thermo data
            if (dup[spec.name] == 2) {
                log << "Warning: more than one THERMO record for "
                    << "species " << spec.name << endl;
                log << "Record at line " << m_line 
                    << " of " << m_ckfilename << " ignored." << endl;
                already_read = true;
		storedSpec=species[spec.name]; // previously stored value
		
		if(!compareSpeciesData(storedSpec,spec,log))
		  {
		    log << "Warning: NASA thermo polynomial record at line "
                        << m_line << endl;
		    log << "         does not match previous definition for "
		        << spec.name << endl;
		  }
		
		
            }
            dup[spec.name] = 2;

            if (!already_read && (getAllSpecies 
                    || (find(names.begin(), names.end(), spec.name) 
                        < names.end())))
            {

                if (spec.tmid == 0.0) {
                    spec.tmid = tmid;
                    log << "Warning: default Tmid used for species " 
                        << spec.name << endl;
                    if (spec.tmid < 0.0) {
                        log << "Error: no default Tmid has been entered!" 
                            << endl;
                    }
                }
                species[spec.name] = spec;

                if (verbose) {        
                    log << endl << "found species " << spec.name;
                    log << " at line " << m_line 
                        << " of " << m_ckfilename;
                    writeSpeciesData(log, spec);
                }
                checkTemps(log, spec.tlow, spec.tmid, spec.thigh);
                if (getAllSpecies) {
                    names.push_back(spec.name);
                    nsp = static_cast<int>(names.size());
                }
                else
                    nsp--;
            }
        }
        return true;
    }


    /**
     *
     * Read one 4-line species definition record in NASA format.
     *
     */

    void CKParser::readThermoRecord(Species& sp) {
        string s;
        string numstr;
        double cf;

        // look for line 1, but if a keyword is found first or the end of
        // the file is reached, return "<END>" as the species name
        string comment;
        do {
            getCKLine(s, comment);
            if (isKeyword(s) || match(s, "<EOF>")) {
                sp.name = "<END>";
                putCKLine(s, comment);
                return;
            }
        }
        while ((s.size() < 80) || (s[79] != '1'));

        // next 4 lines must be the NASA-format lines without intervening
        // comments.

        //------------- line 1 ---------------------------

        if (s[79] != '1') illegalThermoLine(*m_log, s[79], m_line);

        // extract the species name and the id string (date)
        //string nameid = s.substr(0,24);
        string nameid = s.substr(0,16); //First 16 characters are the name
        vector<string> toks;
        getTokens(nameid, static_cast<int>(nameid.size()), toks);
        sp.name = toks[0];
        sp.id = s.substr(16,24); //16-25 characters are the id

//        unsigned int j;
//        for (j = 1; j < toks.size(); j++) {
//	  if (j > 1) sp.id += ' ';  
//            sp.id += toks[j];
//        }

        // Continuation lines defining the element composition in using a 
        // free-form format are indicated with the '&' character appearing 
        // in column 81 of line 1.  Any number of continuation lines may be 
        // included by adding '&' at the end of the previous line. The element
        // counts are treated as integers.
        //printf("# DEBUG: Found first thermo line for species %s on line %d\n",
        //       sp.name.c_str(),m_line);
        bool use_freeform = false;
	string freeform_composition = "";
        if(s.size() >= 81) {
          if(s[80] == '&') {
            // next line is a free-form definition of the element composition
            use_freeform = true;
          }
        }
        if(use_freeform) {
          string continuation_line;
          size_t pos_first_ampersand;

          getCKLine(continuation_line, comment);
          pos_first_ampersand = continuation_line.find_first_of("&");
          // store the composition line excluding the ampersand
          freeform_composition += 
            continuation_line.substr(0,pos_first_ampersand);

          while(pos_first_ampersand != std::string::npos) {
            getCKLine(continuation_line, comment);
            pos_first_ampersand = continuation_line.find_first_of("&");
            // store the composition line excluding the ampersand
            freeform_composition += string(" ") +
              continuation_line.substr(0,pos_first_ampersand);
          }
          //printf("# DEBUG: freeform composition = %s\n", 
          //       freeform_composition.c_str());
        }

        int iloc;
        string elementSym;
        double atoms;
        int i;

        if(use_freeform) {
          vector<string> composition_tokens;
          getTokens(freeform_composition, 
                    static_cast<int>(freeform_composition.size()), 
                                     composition_tokens);
          if(composition_tokens.size()%2 == 1) {
            // incomplete composition specification, must have pairs
            //    "element  quantity"
            *m_log << "# ERROR: Can not process the free-form composition string of species "
                   << sp.name << endl;
            *m_log << "#        composition string = '" 
                   << freeform_composition << "'" << endl;
            throw CK_SyntaxError(*m_log,
                                 "free-form composition string",
                                 m_line);
          }
          for(i=0; i < (int)composition_tokens.size(); i+=2) {
            elementSym = composition_tokens[i];
            atoms = de_atof(composition_tokens[i+1]);
            addElement(elementSym, atoms, sp, *m_log);
          }

          //for(i = 0; i<(int)composition_tokens.size(); ++i) {
          //  printf("# DEBUG: species %s token %d '%s'\n",
          //         sp.name.c_str(),
          //         i,
          //         composition_tokens[i].c_str());
	  //}

          // extract the remaining line 1 data using the same cantera commands 
          // single-character phase descriptor
          sp.phase = s[44];

          // low, high, and mid temperatures
          sp.tlow = de_atof(s.substr(45,10));
          sp.thigh = de_atof(s.substr(55,10));
          if (!m_nasafmt) {
            sp.tmid = de_atof(s.substr(65,8));
          }

        } else {
        
          // elemental composition (first 4 elements)
          for (i = 0; i < 4; i++) {
            elementSym = "";
            iloc = 24 + 5*i;
	    if (s[iloc] != ' ') {
	      if (s[iloc+1] != ' ') elementSym = s.substr(iloc,2);
	      else elementSym = s.substr(iloc,1);
	    }
            else if (s[iloc+1] != ' ') elementSym = s.substr(iloc+1,1);
            atoms = de_atof(s.substr(iloc+2,3));
            addElement(elementSym, atoms, sp, *m_log);
          }

          // single-character phase descriptor
          sp.phase = s[44];

          // low, high, and mid temperatures
          sp.tlow = de_atof(s.substr(45,10));
          sp.thigh = de_atof(s.substr(55,10));

          if (!m_nasafmt) {
            sp.tmid = de_atof(s.substr(65,8));

            // fifth element, if any
            elementSym = "";
            if (s[73] != ' ') elementSym += s[73];
            if (s[74] != ' ') elementSym += s[74];
            atoms = de_atof(s.substr(75,3));
            addElement(elementSym, atoms, sp, *m_log);

            // additional elements, if any
            elementSym = "";
            int loc = 80;
            while (loc < (int) (s.size()-9)) {
                elementSym = "";
                if (s[loc] != ' ') elementSym += s[loc];
                if (s[loc+1] != ' ') elementSym += s[loc+1];
                atoms = de_atof(s.substr(loc+2,8));
                addElement(elementSym, atoms, sp, *m_log);
                loc += 10;
            }
          } // end if(!m_nasafmt)
        } // end of else from if(use_freeform)
        //-------------- line 2 ----------------------------

        getCKLine(s, comment);
        if (s.size() < 80) illegalThermoLineTooShort(*m_log, m_line);
        if (s[79] != '2') illegalThermoLine(*m_log, s[79], m_line);
        for (i = 0; i < 5; i++) {
            numstr = s.substr(i*15, 15);
            cf = getNumberFromString(numstr);
            if (cf == UNDEF) illegalNumber(*m_log, numstr, m_line);
            sp.highCoeffs.push_back(cf);
        }

        //-------------- line 3 ----------------------------

        getCKLine(s, comment);
        if (s.size() < 80) illegalThermoLineTooShort(*m_log, m_line);
        if (s[79] != '3') illegalThermoLine(*m_log, s[79], m_line);
        for (i = 0; i < 2; i++) {
            numstr = s.substr(i*15, 15);
            cf = getNumberFromString(numstr);
            if (cf == UNDEF) illegalNumber(*m_log, numstr, m_line);
            sp.highCoeffs.push_back(cf);
        }
        for (i = 2; i < 5; i++) {
            numstr = s.substr(i*15, 15);
            cf = getNumberFromString(numstr);
            if (cf == UNDEF) illegalNumber(*m_log, numstr, m_line);
            sp.lowCoeffs.push_back(cf);
        }

        //--------------- line 4 ----------------------------

        getCKLine(s, comment);
        if (s.size() < 80) illegalThermoLineTooShort(*m_log, m_line);
        if (s[79] != '4') illegalThermoLine(*m_log, s[79], m_line);
        for (i = 0; i < 4; i++) {
            numstr = s.substr(i*15, 15);
            cf = getNumberFromString(numstr);
            if (cf == UNDEF) illegalNumber(*m_log, numstr, m_line);
            sp.lowCoeffs.push_back(cf);
        }
        sp.valid = 1;
    }


    
    void CKParser::missingAuxData(const string& kw) {
        throw CK_SyntaxError(*m_log, kw + 
            " keyword must be followed by slash-delimited data.", m_line);
    }


    /**
     *  Parse the REACTION section of the input file, and return
     *  a list of Reaction objects and the units.
     */
    bool CKParser::readReactionSection(const vector<string>& speciesNames, 
        vector<string>& elementNames, reactionList& reactions,
        ReactionUnits& units) 
    {
        string s, comment;
        vector<string> toks;
        int nRxns = 0;

        vector<string> rc, pr;
        vector_int c;

    // advance to the beginning of the REACTION section
        do {
            getCKLine(s, comment);
            if (match(s, "<EOF>")) return false;
        }
        while (!match(s,"REAC"));
 

    // look for units specifications

        getTokens(s, static_cast<int>(s.size()), toks);
        string tok;
        units.ActEnergy = Cal_per_Mole;
        units.Quantity = Moles;
        unsigned int ir;
        for (ir = 1; ir < toks.size(); ir++) {
            tok = toks[ir];
            if (match(tok,"CAL/MOLE")) {
                units.ActEnergy = Cal_per_Mole;
            } else if (match(tok,"KCAL/MOLE")) {
                units.ActEnergy = Kcal_per_Mole;
            } else if (match(tok,"JOULES/MOLE")) {
                units.ActEnergy = Joules_per_Mole;
            } else if (match(tok,"KJOULES/MOLE")) {
                units.ActEnergy = Kjoules_per_Mole;
            } else if (match(tok,"KELVINS")) {
                units.ActEnergy = Kelvin;
            } else if (match(tok,"EVOLTS")) {
                units.ActEnergy = Electron_Volts;
                throw CK_SyntaxError(*m_log,
                      " EVOLTS units not supported", m_line);
            } else if (match(tok,"MOLES")) {
                units.Quantity = Moles;
            } else if (match(tok,"MOLECULES")) {
                units.Quantity = Molecules;
                throw CK_SyntaxError(*m_log,
                      " MOLECULES units not supported", m_line);
            }
        }

        Reaction rxn;

        vector<string> cm;
        bool ok = true;
 
        if (debug) {
            *m_log << "CKParser::readReactions  ---> DEBUG MODE" << endl;
        }

        char * split_rev_flag_char;
        int split_rev_flag = 0;
        split_rev_flag_char = getenv("ZERORK_SPLIT_REVERSIBLE_REACTIONS");
        if (split_rev_flag_char!=NULL) {
           split_rev_flag = atoi(split_rev_flag_char);
        }

        while (1 > 0) {

            // skip blank or comment lines
            do {
                getCKLine(s, comment);
                cm.push_back(comment);
            }
            while (s == "" && comment[0] != '%');

#undef DEBUG_LINE
#ifdef DEBUG_LINE
            *m_log << "Line: " << s << endl;
#endif
            // end of REACTION section or EOF
            /// @todo does this handle case of 1 reaction correctly?
            if (isKeyword(s) || s == "<EOF>") {
                if (nRxns > 0) {
                    rxn.number = nRxns;
                    isFinalReactionValid(rxn,
                                         m_log,
                                         m_line);

                    if(split_rev_flag > 0 &&
                           rxn.isReversible && rxn.krev.A >= 0.0) {
                       Reaction frxn = forwardReaction(rxn);
                       reactions.push_back(frxn);
                       nRxns++;
                       Reaction rrxn = reverseReaction(rxn);
                       rrxn.number = nRxns;
                       reactions.push_back(rrxn);
                    } else {
                       reactions.push_back(rxn);
                    }
                    //rxn.comment.clear();
                }
                if (nRxns > 0) return ok;
                return false;
            }

            // rxn line
	    //string::size_type eqloc;
	    int eqloc;
            string sleft, sright;
            bool auxDataLine, metaDataLine;

            
            // if the line contains an '=', it is the start of a new reaction.
            // In this case, add the previous reaction to the output list,
            // increment the number of reactions, and start processing the
            // new reaction.

            eqloc = s.find_first_of("=");
            metaDataLine = false;
            auxDataLine = false;

            // look for a metadata line
            if (s[0] == '%') {
                metaDataLine = true;
                if (eqloc > 0 && eqloc < int(s.size())) {
                    int ierr, ierp;
                    vector<grouplist_t> rg, pg;
                    s[eqloc] = ' ';
                    ierr = getGroups(s.begin(), s.begin() + eqloc, 
                        elementNames, rg);
                    ierp = getGroups(s.begin() + eqloc, s.end(), 
                        elementNames, pg);
                    unsigned int nr = 
			static_cast<unsigned int>(rxn.reactants.size());
                    unsigned int nratoms = 0;
                    for (unsigned int ij = 0; ij < nr; ij++) 
                        nratoms += int(rxn.reactants[ij].number);
                    if (rg.size() != nratoms) 
                        throw CK_SyntaxError(*m_log,
                            " groups not specified for all reactants", m_line);
                    else if (ierr < 0)
                        throw CK_SyntaxError(*m_log,
                            " error in reactant group specification", m_line);
                    for (unsigned int ir = 0; ir < nr; ir++) {
                        rxn.reactants[ir].groups = rg[ir];
                    }
                    unsigned int np = 
			static_cast<unsigned int>(rxn.products.size());
                    unsigned int npatoms = 0;
                    for (unsigned int ik = 0; ik < np; ik++) 
                        npatoms += int(rxn.products[ik].number);
                    if (pg.size() != npatoms) 
                        throw CK_SyntaxError(*m_log,
                            " groups not specified for all products", m_line);
                    else if (ierp < 0)
                        throw CK_SyntaxError(*m_log,
                            " error in product group specification", m_line);
                    for (unsigned int ip = 0; ip < np; ip++) {
                        rxn.products[ip].groups = pg[ip];
                    }             
                }   
            }

            else if (eqloc >= 0 && eqloc < int(s.size())) {
                if (nRxns > 0) {
                    rxn.number = nRxns;
                    isFinalReactionValid(rxn,
                                         m_log,
                                         m_line);
                    if(split_rev_flag > 0 &&
                           rxn.isReversible && rxn.krev.A >= 0.0) {
                       Reaction frxn = forwardReaction(rxn);
                       reactions.push_back(frxn);
                       nRxns++;
                       Reaction rrxn = reverseReaction(rxn);
                       rrxn.number = nRxns;
                       reactions.push_back(rrxn);
                    } else {
                       reactions.push_back(rxn);
                    }
                }
                nRxns++;
                rxn = Reaction();
                rxn.comment = cm;
                cm.clear();
                if (debug) {
                    *m_log << "Parsing reaction " << nRxns << endl;
                }
            }
            else auxDataLine = true;
            if (comment != "") rxn.lines.push_back(s+'!'+comment);
            else rxn.lines.push_back(s);

            if (!auxDataLine && !metaDataLine) {

                // depending on the form of the 'equals' symbol,
                // determine whether the reaction is reversible or
                // irreversible, and separate it into strings for 
                // each side.
	
                if (eqloc = int(s.find("<=>")), eqloc >= 0) {
                    rxn.isReversible = true;
                    sleft = s.substr(0, eqloc);
                    sright = s.substr(eqloc+3,1000);
                }
                else if (eqloc = int(s.find("=>")), eqloc >= 0) {
                    rxn.isReversible = false;
                    sleft = s.substr(0, eqloc);
                    sright = s.substr(eqloc+2,1000);
                }
                else if (eqloc = int(s.find("=")), eqloc >= 0) {
                    rxn.isReversible = true;
                    sleft = s.substr(0, eqloc);
                    sright = s.substr(eqloc+1,1000);
                }
                else throw CK_SyntaxError(*m_log, 
                    "expected <=>, =>, or =", m_line);

                if (debug) {
                    *m_log << s << endl;
                    if (rxn.isReversible) 
                        *m_log << "Reaction is reversible." << endl;
                    else
                        *m_log << "Reaction is irreversible." << endl;
                }

                string::size_type mloc, mloc2;

                // process reactants	   
                if (debug) *m_log << "Processing reactants..." << sleft << endl;
                removeWhiteSpace(sleft);
                if (debug) *m_log << "After removing white space: " 
                                << sleft << endl;
                rxn.isFalloffRxn = false;

                string sm, mspecies;

                mloc = sleft.find("(+");
                if (mloc != string::npos) {
                    sm = sleft.substr(mloc+2, 1000);
                    mloc2 = sm.find(")");
                    if (mloc2 != string::npos) {
                        mspecies = sm.substr(0,mloc2);
                        rxn.isFalloffRxn = true;
                        rxn.type = Falloff;
                        sleft = sleft.substr(0, mloc);
                        if (mspecies == "M" || mspecies == "m") {
                            rxn.thirdBody = "M";
                        }
                        else {
                            rxn.thirdBody = mspecies;
                        }
                        if (debug) {
                            *m_log << "Falloff reaction. Third body = " 
                                 << rxn.thirdBody << endl;
                        }
                    }
                    else throw CK_SyntaxError(*m_log, 
                        "missing )", m_line);
                }

                else if ((mloc = sleft.find("+M"), mloc != string::npos) ||
                    (mloc = sleft.find("+m"), mloc != string::npos)) {

                    if (static_cast<int>(mloc) ==
			static_cast<int>(sleft.size()) - 2) {
                        rxn.isThreeBodyRxn = true;
                        rxn.type = ThreeBody;
                        sleft = sleft.substr(0, mloc);
                        rxn.thirdBody = "M";
                        if (debug) {
                            *m_log << "Three-body reaction." << endl;
                        }
                    }
                    else if (debug) {
                        *m_log << "Reactant string contains +M or +m, but \n"
                             << "not last two characters of string: " 
                             << "\"" << sleft << "\"\n"
                             << "NOT a three-body reaction." << endl;
                    }
                }

                getSpecies(sleft.c_str(),static_cast<int>(sleft.size()),
                    rxn.reactants, debug, *m_log);
                int ir = static_cast<int>(rxn.reactants.size());
                for (int iir = 0; iir < ir; iir++) {
                    if (find(speciesNames.begin(), speciesNames.end(),
                            rxn.reactants[iir].name) >= speciesNames.end()) 
                        throw CK_SyntaxError(*m_log,
                            "undeclared reactant species "
                            +rxn.reactants[iir].name, m_line);
                }
        

                // process Arrhenius coefficients
                getTokens(sright, static_cast<int>(sright.size()), toks);
                int ntoks = static_cast<int>(toks.size());
                if (ntoks < 3) {
                    throw CK_SyntaxError(*m_log, 
                        "expected 3 Arrhenius parameters", m_line);
                }
                rxn.kf.A = de_atof(toks[ntoks - 3]);
                rxn.kf.n = de_atof(toks[ntoks - 2]);
                rxn.kf.E = de_atof(toks[ntoks - 1]);

                // 2/10/03: allow negative prefactor but print a warning
		// 3/06/24: don't allow negative prefactor
                if (rxn.kf.A < 0.0)
                //    *m_log << "Warning: negative prefactor at line "
                //           << m_line << endl;
                    throw CK_SyntaxError(*m_log, "negative prefactor", m_line);

                if (debug) *m_log << "Processing products..." << sright << endl;
                sright = "";
		for(int i=0; i<ntoks-3;++i) {
			sright += toks[i];
		}
                if (debug) *m_log << "After removing Arrhenius parameters, "
                                << "\nproduct string = " << sright << endl;

                removeWhiteSpace(sright);
                if (debug) *m_log << "After removing white space: " 
                                << sright << endl;
                mloc = sright.find("(+");
                if (mloc != string::npos) {
                    sm = sright.substr(mloc+2, 1000);
                    mloc2 = sm.find(")");
                    if (mloc2 != string::npos) {
                        mspecies = sm.substr(0,mloc2);

                        if (rxn.type == ThreeBody) 
                            throw CK_SyntaxError(*m_log, 
                                "mismatched +M or (+M)", m_line);

                        rxn.isFalloffRxn = true;
                        rxn.type = Falloff;
                        if (debug) {
                            *m_log << "Falloff reaction. Third body = " 
                                 << rxn.thirdBody << endl;
                        }
                    }
                    else throw CK_SyntaxError(*m_log, 
                        "missing )", m_line);

                    sright = sright.substr(0, mloc);

                    if (mspecies == "M" || mspecies == "m") {
                        rxn.thirdBody = "M";
                    }
                    else {
                        if (rxn.thirdBody != mspecies) 
                            throw CK_SyntaxError(*m_log, 
                                "mismatched third body", m_line);
                        rxn.thirdBody = mspecies;
                    }
                }

                else if ((mloc = sright.find("+M"), mloc != string::npos) ||
                    (mloc = sright.find("+m"), mloc != string::npos)) {

                    if (static_cast<int>(mloc) == 
			static_cast<int>(sright.size()) - 2) {

                        if (rxn.type == Falloff) 
                            throw CK_SyntaxError(*m_log, 
                                "mismatched +M or (+M)", m_line);
                        rxn.isThreeBodyRxn = true;
                        rxn.thirdBody = "M";
                        sright = sright.substr(0, mloc);
                        if (debug) {
                            *m_log << "Three-body reaction." << endl;
                        }
                    }
                    else if (debug) {
                        *m_log << "Product string contains +M or +m, but \n"
                             << "not last two characters of string: " 
                             << "\"" << sright << "\"\n"
                             << "NOT a three-body reaction." << endl;
                    }		
                }
                getSpecies(sright.c_str(),static_cast<int>(sright.size()),
                    rxn.products, debug, *m_log);
                int ip = static_cast<int>(rxn.products.size());
                for (int iip = 0; iip < ip; iip++) {
                    if (find(speciesNames.begin(), speciesNames.end(),
                            rxn.products[iip].name) >= speciesNames.end()) 
                        throw CK_SyntaxError(*m_log,
                            "undeclared product species "+rxn.products[iip].name, m_line);
                }
            }

            // auxiliary data line
            else if (auxDataLine) {

                bool hasAuxData;
                string name, data;
                map<string, int> kwindex;
                while (1 > 0) {

                    hasAuxData = extractSlashData(s, name, data);
                    if (!hasAuxData && name == "") break;

                    // check for duplicate keyword
                    if (kwindex[name]) {
                        throw CK_SyntaxError(*m_log, 
                            "duplicate auxiliary data keyword " + name, m_line);
                    }
                    else 
                        kwindex[name] = 1;


                    // low-pressure rate coefficient for falloff rxn

                    if (match(name,"LOW")) {
                        vector<string> klow;
                        rxn.type = Falloff;
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), klow);
                            if (klow.size() != 3) {
                                throw CK_SyntaxError(*m_log, 
                                    "expected 3 low-pressure Arrhenius parameters", m_line);
                            }
                            rxn.kf_aux.A = de_atof(klow[0]);
                            rxn.kf_aux.n = de_atof(klow[1]);
                            rxn.kf_aux.E = de_atof(klow[2]);
                        }
                        else 
                            missingAuxData("LOW");
                    }


                    // falloff parameters

                    else if (match(name,"TROE")) {
                        vector<string> falloff;
                        if (kwindex["SRI"] > 0)
                        {
                            throw CK_SyntaxError(*m_log, 
                                "cannot specify both SRI and TROE", m_line);
                        }

                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), falloff);
                            int nf = static_cast<int>(falloff.size());
                            double ff;
                            rxn.falloffType = Troe;
                            for (int jf = 0; jf < nf; jf++) {
                                ff = de_atof(falloff[jf]);
                                rxn.falloffParameters.push_back(ff);
                            }
                        }
                        else 
                            missingAuxData("TROE");
                    }

                    else if (match(name,"SRI")) {
                        vector<string> falloff;
                        if (kwindex["TROE"] > 0) {
                            throw CK_SyntaxError(*m_log, 
                                "cannot specify both SRI and TROE", m_line);
                        }
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), falloff);
                            int nf = static_cast<int>(falloff.size());
                            rxn.falloffType = SRI;
                            double ff;
                            for (int jf = 0; jf < nf; jf++) {
                                ff = de_atof(falloff[jf]);
                                rxn.falloffParameters.push_back(ff);
                            }
                        }
                        else missingAuxData("SRI");
                    }



                    // reverse rate coefficient 

                    else if (match(name,"REV")) {
                        vector<string> krev;
                        if (!rxn.isReversible) {
                            throw CK_SyntaxError(*m_log, 
                                "reverse rate parameters can only be "
                                "specified for reversible reactions", m_line);
                        }
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), krev);
                            if (krev.size() != 3) {
                                throw CK_SyntaxError(*m_log, 
                                    "expected 3 Arrhenius parameters", m_line);
                            }                            
                            rxn.krev.A = de_atof(krev[0]);
                            rxn.krev.n = de_atof(krev[1]);
                            rxn.krev.E = de_atof(krev[2]);
                        }
                        else 
                            missingAuxData("REV");
                    }


                    else if (match(name,"DUP"))
                        rxn.isDuplicate = true;

                    else if (match(name,"END")) {
                        string c = "";
                        putCKLine(name,c);
                        break;
                    }


                    // Landau-Teller reaction rate parameters

                    else if (match(name,"LT")) {
                        vector<string> bc;
                        rxn.kf.type = LandauTeller;
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), bc);
                            rxn.kf.B = de_atof(bc[0]);
                            rxn.kf.C = de_atof(bc[1]);
                        }
                        else 
                            missingAuxData("LT");
                    }

                    else if (match(name,"RLT")) {
                        vector<string> bc;
                        rxn.krev.type = LandauTeller;
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), bc);
                            rxn.krev.B = de_atof(bc[0]);
                            rxn.krev.C = de_atof(bc[1]);
                        }
                        else 
                            missingAuxData("RLT");
                    }


                    // chem activation reactions
                    else if (match(name,"HIGH")) {
                        vector<string> khigh;
                        rxn.type = ChemAct;
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), khigh);
                            rxn.kf_aux.A = de_atof(khigh[0]);
                            rxn.kf_aux.n = de_atof(khigh[1]);
                            rxn.kf_aux.E = de_atof(khigh[2]);
                        }
                        else 
                            missingAuxData("HIGH");
                    }
                    // FORD reaction
                    else if (match(name,"FORD")) {
                        //throw CK_SyntaxError(*m_log, "FORD reaction type not supported", m_line);
                        //rxn.type = RealOrder;
                        rxn.isRealOrder = true;
                        vector<string> nmord;
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()), 
                                nmord);
                            rxn.fwdOrder[nmord[0]] = de_atof(nmord[1]);
                        }
                        else
                            missingAuxData("FORD");
                    }
                    else if (match(name,"RORD")) {
                        //rxn.type = RealOrder;
                        rxn.isRealOrder = true;
                        vector<string> nmord;
                        if (hasAuxData) {
                            getTokens(data, static_cast<int>(data.size()),  
                                nmord);
                            rxn.revOrder[nmord[0]] = de_atof(nmord[1]);
                        }
                        else
                            missingAuxData("RORD");
                    }


                    // PLOG reaction
                    else if(match(name,"PLOG")) {
                      vector<string> kplog;
                      rxn.type = PLogInterpolation;
                      rxn.kf.type = PLogInterpolation;
                      if(hasAuxData) {
                        getTokens(data,
                                  static_cast<int>(data.size()),
                                  kplog);
                        if (kplog.size() != 4) {
                          throw CK_SyntaxError(*m_log, 
                            "expected 4 PLOG parameters", m_line);
                        }
                        if(de_atof(kplog[0])<0.0) {
                          throw CK_SyntaxError(*m_log,
                            "PLOG pressures are required to be >= 0.0", m_line);
                        }
                        rxn.kf.pres_pts_plog.push_back(de_atof(kplog[0]));
                        rxn.kf.A_plog.push_back(de_atof(kplog[1]));
                        rxn.kf.n_plog.push_back(de_atof(kplog[2]));
                        rxn.kf.E_plog.push_back(de_atof(kplog[3]));
                        rxn.isPLogInterpolation = true;
                        //printf("Found PLOG at p = %6.2f atm\n",
                        //       de_atof(kplog[0]));
                        
                      } else {
                        missingAuxData("PLOG");
                      }
                    } // end of PLOG reaction     
                    else if (find(speciesNames.begin(), speciesNames.end(), name) 
                        < speciesNames.end()) {
                        if (hasAuxData) {
                            if (rxn.thirdBody == name || rxn.thirdBody == "M")
                                rxn.e3b[name] = de_atof(data);
                            else if (rxn.thirdBody == "<none>") {
                                *m_log << "Error in reaction " << nRxns 
                                       << ": third-body collision efficiencies cannot be specified"
                                       << " for this reaction type." << endl;
                                throw CK_SyntaxError(*m_log, 
                                    "third-body efficiency error", m_line);
                            }
                            else {
                                *m_log << "Reaction " << nRxns << ": illegal species in enhanced "
                                       << "efficiency specification. Species = " 
                                       << name << " rxn.thirdBody = " 
                                       << rxn.thirdBody << endl; 
                                throw CK_SyntaxError(*m_log, 
                                    "third-body efficiency error", m_line);
                            }
                        }
                        else missingAuxData(name);
                    }
                    else {
                        Reaction::auxdata vals;
                        vector<string> toks;
                        getTokens(data, static_cast<int>(data.size()), toks);
                        int ntoks = static_cast<int>(toks.size());
                        for (int itok = 0; itok < ntoks; itok++) {
                            vals.push_back(de_atof(toks[itok]));
                        }
                        rxn.otherAuxData[name] = vals;
                    }
                }
            }
        }
        return false;
    }



    int parseGroupString(string str, vector<string>& esyms, group_t& result) {
        bool inSymbol=true;
        string s = str + '-';
        int i;
        string num, sym;
        int eindx;
        string::const_iterator begin = s.begin();
        string::const_iterator end = s.end();
        vector<string>::iterator e;
        result.resize(static_cast<size_t>(esyms.size()),0);
        for (; begin != end; ++begin) {

            // new element
            if (*begin == '-') {
                e = find(esyms.begin(), esyms.end(), sym);
                if (e == esyms.end()) return -1;
                eindx = static_cast<int>(e - esyms.begin());
                if (num != "") 
                    i = atoi(num.c_str());
                else 
                    i = 1;
                result[eindx] = i;
                sym = "";
                num = "";
                inSymbol = true;
            }
            else if (isdigit(*begin)) {
                inSymbol = false;
                num += *begin;
            }
            else if (isalpha(*begin) && inSymbol) {
                sym += *begin;
            }
        }
        return 1;
    }

  bool CKParser::compareSpeciesData(Species &a, Species &b, ostream &log)
  {
    
    if(a.tmid != b.tmid)
      {return false;}
    if(a.lowCoeffs.size() != 7)
      {
	log << "WARNING: Species X thermodynamics has " 
	    << a.lowCoeffs.size() << " low coefficients" << endl;
	log << "         in bool CKParser::compareSpeciesData(Species &X, Species &Y)" << endl;
	return false;
      }
    if(b.lowCoeffs.size() != 7)
      {
	log << "WARNING: Species Y thermodynamics has " 
	    << a.lowCoeffs.size() << " low coefficients" << endl;
	log << "         in bool CKParser::compareSpeciesData(Species &X, Species &Y)" << endl;
	return false;
      }
    if(a.highCoeffs.size() != 7)
      {
	log << "WARNING: Species X thermodynamics has " 
	    << a.lowCoeffs.size() << " high coefficients" << endl;
	log << "         in bool CKParser::compareSpeciesData(Species &X, Species &Y)" << endl;
	return false;
      }
    if(b.highCoeffs.size() != 7)
      {
	log << "WARNING: Species Y thermodynamics has " 
	    << a.lowCoeffs.size() << " high coefficients" << endl;
	log << "         in bool CKParser::compareSpeciesData(Species &X, Species &Y)" << endl;
	return false;
      }

    if(a.lowCoeffs[0] != b.lowCoeffs[0])
      {return false;}
    if(a.lowCoeffs[1] != b.lowCoeffs[1])
      {return false;}
    if(a.lowCoeffs[2] != b.lowCoeffs[2])
      {return false;}
    if(a.lowCoeffs[3] != b.lowCoeffs[3])
      {return false;}
    if(a.lowCoeffs[4] != b.lowCoeffs[4])
      {return false;}
    if(a.lowCoeffs[5] != b.lowCoeffs[5])
      {return false;}
    if(a.lowCoeffs[6] != b.lowCoeffs[6])
      {return false;}
 
    if(a.highCoeffs[0] != b.highCoeffs[0])
      {return false;}
    if(a.highCoeffs[1] != b.highCoeffs[1])
      {return false;}
    if(a.highCoeffs[2] != b.highCoeffs[2])
      {return false;}
    if(a.highCoeffs[3] != b.highCoeffs[3])
      {return false;}
    if(a.highCoeffs[4] != b.highCoeffs[4])
      {return false;}
    if(a.highCoeffs[5] != b.highCoeffs[5])
      {return false;}
    if(a.highCoeffs[6] != b.highCoeffs[6])
      {return false;}
    
    return true;
  }

}  // ckr namespace


    


