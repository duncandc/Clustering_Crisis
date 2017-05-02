#!/usr/local/bin/perl -- # -*-Perl-*-

######################################################################
#
# nat2jour.pl
#
# Program for converting tex/bbl files generated using the natbib
# (v5.3) BibTeX style into files which may be submitted electronically
# to astronomy journals.
#
# Written by Jonathan Baker <jbaker@astro.berkeley.edu>
# inspired by 2apj.pl code by Eddie Baron
#
# $Log: nat2jour.pl,v $
# Revision 1.6  2000/04/21 17:48:02  jbaker
# Changed addition of space before \cite's so that it only happens when
# \cite is preceeded by a \... command (so that it doesn't happen with
# ~\cite, for example, as suggested by Norman Gray).
#
# Revision 1.5  2000/04/20 22:16:26  jbaker
# If using -references, insert a space before \cite in case it is
# terminating a command (like \protect, for example).
#
# Revision 1.4  1999/09/01 02:45:19  jbaker
# Added flag to generate `references' environment instead of `thebibliography'.
#
# Revision 1.3  1998/08/30 19:48:32  jbaker
# Added RCS keywords.  Non-MNRAS lettering style changed from "1998a,b" to
# "1998a, 1998b".
#
#
######################################################################

$MaxAuths = 8;
$inline = 0;
$refenv = 0; $bibenv = "thebibliography";
$etal = "{et al.}";

$usage = <<EOT;

Usage: $0 [options] INPUT [OUTPUT]

$Id: nat2jour.pl,v 1.6 2000/04/21 17:48:02 jbaker Exp $

Convert BibTeX/Natbib tex/bbl files into files which may be submitted
to astronomy journals.  Inlines citations adding \\markcite or
\\nocite and formats \\bibitems and removes natbib package dependencies,
stripping "%nat" comments.  Also implements astronomy 3-author citation
style and truncates long author lists.

Arguments:
  INPUT        If this is "foo", input files are "foo.tex" and "foo.bbl"
  OUTPUT       (optional) Same for output files; default appends "-aas".

Options:
  -inline      Inline the bibliography into a single LaTeX file
  -maxauth     Set maximum number of authors before truncation 
                 (8=default, 0=no limit)
  -references  Instead of a "thebibliography" environment, create a
                 "references" environment.  Also leaves out \markcite
                 commands.
  -help        Print this message and exit successfully

EOT

# Options
while (@ARGV && $ARGV[0] =~ /^-/) {
  $_ = shift(@ARGV);

  if (/^-i(nline)?$/) { $inline = 1; next; }            # -inline

  if (/^-m(axauth)?=?(\d+)?$/) {                        # -maxauth
    $MaxAuths = $2 || shift(@ARGV);
    die("$0: option -maxauth requires integer argument\n") 
      if ($MaxAuths !~ /^\d+$/);
    warn("** Lettering may be incorrect for -maxauth=$MaxAuths.\n")
      if ($MaxAuths > 0 && $MaxAuths < 3);
    next;
  }

  if (/^-r(eferences)?$/) {                             # -references
    $refenv = 1; 
    $bibenv = "references";
    next; 
  }

  if (/^-h(elp)$/) { print $usage; exit; }              # -help

  die $usage;
}

# Arguments
die $usage unless ($#ARGV == 0 || $#ARGV == 1);
$OldBibFile = $ARGV[0] . ".bbl";           # input file names
$OldTexFile = $ARGV[0] . ".tex";
if ($#ARGV == 0 || $ARGV[0] eq $ARGV[1]) { # default output file name
  $OutputRoot = $ARGV[0] . "-aas";
}
else {                                     # use argument
  $OutputRoot = $ARGV[1];
}
  
# Open input files
open(OLD_BIB, "$OldBibFile") || die("Cannot open input file $OldBibFile!\n");
open(OLD_TEX, "$OldTexFile") || die("Cannot open input file $OldTexFile!\n");

# Open output files
$BibFile = $OutputRoot . ".bbl";
$TexFile = $OutputRoot . ".tex";
open(BIB, ">$BibFile") || die("Cannot open output file $BibFile!\n");
open(TEX, ">$TexFile") || die("Cannot open output file $TexFile!\n");

# Write comments to output files
$line = 
  "\n%% --------------------------------------------------------------------";
$Comment = 
  "\n%% " . (scalar localtime) .
  "\n%%   This file was generated automagically from the files" .
  "\n%%   $OldBibFile and $OldTexFile using" .
  "\n%%     $0";
$TexComment = 
  "\n%%   All citations have been inlined and dependencies on the natbib" .
  "\n%%   package have been removed so that this file (together with" .
  "\n%%   $BibFile) should be suitable for submission to journals with" .
  "\n%%   the citation styles of ApJ or MNRAS.";
$BibComment = 
  "\n%%   This file should accompany $TexFile.";
print BIB $line . $Comment . $BibComment . $line . "\n\n";
print TEX $line . $Comment . $TexComment . $line . "\n\n";

#
# Store all the \bibitem data.  Assumes \bibitem's in the input file are 
# terminated by empty lines and that no more than one \bibitem is found
# on a single line.  Expected format is:
#   \bibitem[{short_author_list(year)long_author_list}]{KEY}...
# Note the match will have problems if there are any ']' characters in
# the bibliographic entry, and other deviations from the exact format
# may wreak havoc.
#
print "Reading bib file $OldBibFile... ";
$nBib = 0;
$found = 0;
while (<OLD_BIB>) {
  if (/\\bibitem/) {              # got a new one
    $nBib++;
    $found = 1;
  }
  next if not $found;
  $item .= $_;                    # add line to the current item
  if (/^\n/) {                    # finished reading, now process
    $item =~ s/\n//g;                                  # remove newlines
    $item =~ s/\{\\natexlab\{(.*?)\}\}/\1/g;           # remove \natexlab's
    $item =~ /.*?\[\{(.*?)\((\d{4}[a-z]*)\)(.*?)\}\]\{(.*?)\}(.*)/;
    ($ShortAuth, $year, $LongAuth, $key, $ref) = ($1, $2, $3, $4, $5);
    $LongAuth = $ShortAuth if (length($LongAuth) < 1); # no long author list
    push @keys, $key;
    $Years{$key} = $year;                              # store for later use
    $LongAuths{$key} = $LongAuth;
    $ShortAuths{$key} = $ShortAuth;
    $Refs{$key} = $ref;
    $found = 0;
    $item = "";
  }
}
close(OLD_BIB);
warn("\n** No bibitems found in $OldBibFile!\n") if $nBib < 1;
print "done.\n";

# Make new TeX file with inlined references
print "Writing $TexFile... ";
$mnras = 0;
$preamble = 1;
while (<OLD_TEX>) {
  $line = $_;
  next if (/^\s*\%+nat.*/);                      # remove %nat comments
  $line =~ s/\s*\%+nat.*//;
  if ($line =~ /.*?(%)?.*?\\document(style|class).*?\{(.*?)\}/) {
    if (length($1) <= 0) {                       # not a comment
      if ($3 =~ /mn/) {                          # \document..{..mn..}
	print "looks like an MNRAS file... ";
	$mnras = 1; 
	$line =~ s/mn-nat/mn/;                      # change mn-nat to mn
	$markcite = "\\nocite";                     # use "\nocite"
      }
      else {                                     # otherwise assume AAS
	print "looks like an AAS file... ";
	$markcite = "\\markcite";                   # use "\markcite"
      }
    }
  }
  $line =~ s/\\bibliography\{.*?\}/\\bibliography\{\}/;  # \bibliography
  $line =~ s/\\bibliographystyle/\%\\bibliographystyle/; # \bibliographystyle
  $preamble = 0 if (/\\begin.document./);
  if ($preamble) {                          # in the preamble...
    $line =~ s/^\\citestyle/%\\citestyle/;      # comment out \citestyle
    $line =~ s/,natbib(209)?//;                 # remove ,natbib
    $line =~ s/natbib(209)?,//;                 # remove natbib,
    $line =~ s/(\\usepackage\{natbib(209)?\})/%\1/;
    $line =~ s/\[natbib(209)?\]/\[\]/;          # [natbib] -> []
    print TEX $line;
    next;
  }
  $line =~ s/\\citetext\{(.*?)\}/\1/g;      # \citetext's
  # separate preceding commands by space (can't recognise catcode changes)
  $line =~ s/(\\[a-zA-Z]+)\\cite/\1 \\cite/g if ($refenv);
  $lookfor = "\\cite";  
  if (index($line, $lookfor) == -1) {       # no \cite in this line
    print TEX $line;
    next;
  }
  $pos = -1;
  while (($pos = index($line, $lookfor, $pos)) > -1) {
    $beg = $pos;                            # beginning of \cite*{...}
    $end = index($line, "\}", $beg);        # end
    while ($end <= $beg) {                  # \cite spans more than one line
      $line .= <OLD_TEX>;
      $end = index($line, "\}", $beg);
    }
    $IsComment = index($line, '%', 0);      # \cite is commented out?
    $IsComment = ($IsComment < $pos and $IsComment > -1);
    $cite = substr($line, $beg, $end-$beg+1);        # limit to current \cite
    $cite =~ /\\cite(.*?)(\[(.*?)\])?(\[(.*?)\])?\{(.*)\}/s;
    ($comm, $opt1, $opt2) = ("cite$1", $3, $5);      # command and option text
    $nOpts = (length($2) > 0) + (length($4) > 0);    # number of options
    $KeyList = $6;
    @CiteKeys = split(/\s*,\s*/, $KeyList);          # get cite keys
    $NoYear = ($comm =~ /citeauthor/);               # interpret the command
    $NoAuthor = ($comm =~ /citeyear/);
    $UseLong = ($comm =~ /\*/);
    $NoParen = ($comm =~ /citea/ || $comm eq "citeyear");
    $FullParen = ($comm =~ /citep/);
    $YearParen = not ($NoParen || $FullParen);
    $CommaSep = ($comm =~ /citealp/);
    $OldAuth = $OldYear = $NewCite = "";    # initialize
    $i = 0;
    foreach $CiteKey (@CiteKeys) {          # loop over keys
      $nAuth = 1 + ($LongAuths{$CiteKey} =~ tr/,/,/); # count authors
      if (($UseLong > 0) || ($nAuth==3 and not $cited{$CiteKey})) {
	$auth = $LongAuths{$CiteKey};       # use long form
      }
      else {                               
	$auth = $ShortAuths{$CiteKey};      # use short form
      }
      $cited{$CiteKey} = 1;
      $yearl = $Years{$CiteKey};            # year + letter
      $yearl =~ /(\d+)([a-z]*)/; 
      ($year, $letter) = ($1, $2);
      if ($i > 0 and $auth eq $OldAuth) {   # don't repeat same author name
	if (not $NoYear) {
	  $NewCite = substr($NewCite, 0, -1) if ($YearParen); # remove ')'
	  if ($year eq $OldYear and $mnras) {
	    $NewCite .= ",$letter";         # don't repeat same year
	  }
	  else {
	    $NewCite .= ", $yearl";
	  }
	  $NewCite .= ")" if ($YearParen);                    # restore ')'
	}
      }
      else {
	$NewCite .= "; " if ($i > 0);       # add to previous
	$NewCite .= "$auth" if (not $NoAuthor);
	if (not $NoYear) {
	  $NewCite .= "," if ($CommaSep);
	  $NewCite .= ($YearParen) ? " ($yearl)" : " $yearl";   # year
	}
      }
      $OldAuth = $auth;
      $OldYear = $year;
      $i++;
    }
    if ($nOpts == 1 and length($opt1) > 0) {       # add option text
      $NewCite = substr($NewCite, 0, -1) if ($YearParen);
      $NewCite = "$NewCite, $opt1";
      $NewCite .= ")" if ($YearParen);
    }
    elsif ($nOpts == 2) {
      $NewCite = "$opt1 $NewCite" if (length($opt1) > 0);
      if (length($opt2) > 0) {
	$NewCite = substr($NewCite, 0, -1) if ($YearParen);
	$NewCite = "$NewCite, $opt2";
	$NewCite .= ")" if ($YearParen);
      }
    }
    $NewCite = "($NewCite)" if ($FullParen);
    $NewCite =~ s/,\s*\\&/ \\&/ if ($mnras);
    $NewCite = "$markcite\{$KeyList\}$NewCite" unless ($refenv);
    $NewCite =~ s/\n/\n%/g if $IsComment;  # careful with comments
    substr($line, $beg, $end-$beg+1) = $NewCite;   # insert into line
    $pos++;
  }
  print TEX $line;
}
print "done.\n";
close(TEX);

#
# Make new .bbl bibliography file, looping over items.  
# Output format is \bibitem[long_author_list year]{key}reference
#
print "Writing $BibFile... ";
print BIB "\\begin{$bibenv}";
print BIB "{}" unless ($refenv);
print BIB "\n\n";
foreach $key (@keys) {
  $ref = $Refs{$key};
  $ref =~ s/,(\s*\(.*?\)\s*)$/\1/; # get rid of "," before parenthesized note
  $nAuths = 1 + ($LongAuths{$key} =~ tr/,/,/);   # count authors
  if ($MaxAuths > 0 && $nAuths > $MaxAuths) {    # too many to list
    if ($ref !~ /(.*?)(,)?\s*(\d{4})/) {
      warn("**Can't find year in reference:\n$ref\n");
    }
    ($auths, $lastComma) = ($1, $2);
    $end = index($ref, $3);
    if ($auths =~ /\w/) {        # this avoids refs like "---, 1998, ..."
      $nCommas = $MaxAuths;        # Jones J.J., ...
      $nCommas *= 2 if not $mnras; # Jones, J.J., ...
      if ($nCommas > ($auths =~ tr/,/,/)) {
	warn("**Error trying to truncate author list:\n$ref\n");
      }
      else {
	$pos = -1;
	for $i (1..$nCommas) {     # find the n'th comma
	  $pos = index($auths, ',', $pos);
	  $pos++;
	}
	substr($ref, $pos, $end-$pos-1) = " $etal$lastComma"
      }
    }
  }
  if ($refenv) {
    print BIB "\\reference ";
  }
  else {
    print BIB "\\bibitem[$LongAuths{$key} $Years{$key}]{$key}\n";
  }
  print BIB "$ref\n\n";
}
print BIB "\\end{$bibenv}\n";
close(BIB);
print "done.\n";

#
# Inline the bbl into the tex file
#
if ($inline) {
  $TmpFile = "$TexFile" . "tmp";
  print "Inlining bbl file... ";
  open(TEX, "$TexFile") || die("Can't open file $TexFile!\n");
  open(TMP, ">$TmpFile") || die("Can't write file $TmpFile!\n");
  open(BIB, "$BibFile") || die("Can't open file $BibFile!\n");
  while (<TEX>) {
    $line = $_;
    if ($line !~ /(.*?)(\\bibliography)\{\s*\}(.*)/ || $1 =~ /%/) {
      print TMP $line;
      next;
    }
    print TMP "$1%% $2\n";
    print TMP $3 if ($3 !~ /^\s*\n$/);
    $found = 0;
    while (<BIB>) {
      $found = 1 if ((/\\begin{thebibliography}/) or (/\\begin{references}/));
      print TMP if ($found);
    }
  }
  close(TMP);
  close(TEX);
  close(BIB);
  rename($TmpFile, $TexFile) || die("Can't rename $TmpFile!\n");
  open(BIB, ">>$BibFile");     # touch
  close(BIB);
  print "done.\n";
}  